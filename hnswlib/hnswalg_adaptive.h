#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <vector>

#include "hnswalg.h"

namespace hnswlib {
template <typename dist_t>
class HierarchicalNSWAdaptive : public HierarchicalNSW<dist_t> {
public:
    using Base = HierarchicalNSW<dist_t>;
    using typename Base::CompareByFirst;
    struct AdaptiveConfig {
        size_t sample_k = 16;
        size_t probe_descend_levels = 1;

        float w_density = 0.4f;
        float w_variance = 0.4f;
        float w_skewness = 0.2f;

        float ml_scale_min = 1.0f;
        float ml_scale_max = 1.2f;

        float ef_c_density_gain = 0.2f;
        float ef_c_variance_gain = 0.1f;
        float ef_c_skewness_gain = 0.1f;
        float ef_c_frontier_boost = 0.2f;

        float angular_thresh_base = 0.98f;
        float angular_thresh_delta = 0.10f;
        size_t angular_max_dim = 64;

        float ef_s_scale_min = 1.0f;
        float ef_s_scale_max = 2.5f;
        float ef_s_skewness_boost = 0.5f;

        float ema_alpha = 0.005f;
        size_t ema_warmup = 50;

        float frontier_density_max = 0.35f;
        float frontier_variance_min = 0.70f;

        size_t level_cap_bootstrap = 100;
    };

    AdaptiveConfig cfg;

    HierarchicalNSWAdaptive(SpaceInterface<dist_t>* s, size_t max_elements,
                            size_t M = 16, size_t ef_construction = 200,
                            size_t random_seed = 100,
                            bool allow_replace_deleted = false,
                            bool normalize = false,
                            bool persist_on_write = false,
                            const std::string& persist_location = "")
    : Base(s, max_elements, M, ef_construction, random_seed,
            allow_replace_deleted, normalize, persist_on_write,
            persist_location),
            ema_mean_dist_(0.0f),
            ema_initialized_(false) {}

    struct AdaptiveState {
        float density_score = 0.5f;
        float variance_score = 0.5f;
        float skewness_score = 0.0f;
        float difficulty = 0.5f;
        bool is_frontier = false;
        bool valid = false;

        float revSizeScale(const AdaptiveConfig& c) const {
            if (!valid) return 1.0f;
            float t = 1.0f - density_score; 
            return c.ml_scale_min + t * (c.ml_scale_max - c.ml_scale_min);
        }

        float efConstructionScale(const AdaptiveConfig& c) const {
            if (!valid) return 1.0f;
            float scale = 1.0f + c.ef_c_density_gain * density_score +
                        c.ef_c_variance_gain * variance_score +
                        c.ef_c_skewness_gain * skewness_score;
            if (is_frontier) scale += c.ef_c_frontier_boost;
            return scale;
        }

        float efConstructionScaleLayer0(const AdaptiveConfig& c) const {
            float base = efConstructionScale(c);
            if (is_frontier) base += 0.2f;
            return base;
        }

        float angularThreshold(const AdaptiveConfig& c) const {
            return c.angular_thresh_base - c.angular_thresh_delta * density_score;
        }

        float efSearchScale(const AdaptiveConfig& c) const {
            if (!valid) return 1.0f;
            float base_scale = c.ef_s_scale_min + difficulty * (c.ef_s_scale_max - c.ef_s_scale_min);
            float skew_boost = c.ef_s_skewness_boost * std::tanh(skewness_score * 2.0f);
            return std::min(c.ef_s_scale_max + c.ef_s_skewness_boost, base_scale + skew_boost);
        }
    };

    void addPoint(const void* data_point, labeltype label, bool replace_deleted = false) override {
        {
            std::unique_lock<std::mutex> lock_table(this->label_lookup_lock);
            auto it = this->label_lookup_.find(label);
            if (it != this->label_lookup_.end()) {
                lock_table.unlock();
                Base::addPoint(data_point, label, replace_deleted);
                return;
            }
        }

        tableint cur_c = 0;
        {
            std::unique_lock<std::mutex> lock_table(this->label_lookup_lock);
            if (this->cur_element_count >= this->max_elements_)
                throw std::runtime_error("The number of elements exceeds the specified limit");
            cur_c = this->cur_element_count++;
            this->label_lookup_[label] = cur_c;
        }

        std::unique_lock<std::mutex> lock_el(this->link_list_locks_[cur_c]);

        AdaptiveState state;

        AdaptiveState state_M1;
        if (cfg.ml_scale_max > 1.0f && (signed)this->enterpoint_node_ != -1 && this->maxlevel_ > 0) {
            tableint probe_obj = this->enterpoint_node_;
            dist_t probe_dist = this->fstdistfunc_(data_point, this->getDataByInternalId(probe_obj), this->dist_func_param_);

            int top_lev = this->maxlevel_;
            int stop_lev = std::max(0, top_lev - (int)cfg.probe_descend_levels);
            int cur_lev = top_lev;

            for (; cur_lev > stop_lev; cur_lev--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int* d = this->get_linklist(probe_obj, cur_lev);
                    int sz = this->getListCount(d);
                    auto* dl = (tableint*)(d + 1);
                    for (int i = 0; i < sz; i++) {
                        dist_t dd = this->fstdistfunc_(
                            data_point, this->getDataByInternalId(dl[i]), this->dist_func_param_);
                        if (dd < probe_dist) {
                            probe_dist = dd;
                            probe_obj = dl[i];
                            changed = true;
                        }
                    }
                }
            }

            std::vector<dist_t> quick_samples;
            quick_samples.push_back(probe_dist);

            unsigned int* local_links = (cur_lev == 0) ? this->get_linklist0(probe_obj) : this->get_linklist(probe_obj, cur_lev);
            int lsz = this->getListCount(local_links);
            auto* ldl = (tableint*)(local_links + 1);
            for (int i = 0; i < lsz && quick_samples.size() <= cfg.sample_k; i++)
                quick_samples.push_back(this->fstdistfunc_(data_point, this->getDataByInternalId(ldl[i]), this->dist_func_param_));

            state_M1 = buildState(quick_samples);
        }
        int curlevel =
            sampleAdaptiveLevel(state_M1.valid ? state_M1 : AdaptiveState{});
        this->element_levels_[cur_c] = curlevel;

        std::unique_lock<std::mutex> templock(this->global);
        int maxlevelcopy = this->maxlevel_;
        if (curlevel <= maxlevelcopy) templock.unlock();

        tableint currObj = this->enterpoint_node_;
        tableint enterpoint_copy = this->enterpoint_node_;

        memset(this->data_level0_memory_ + cur_c * this->size_data_per_element_ + this->offsetLevel0_,
                0, this->size_data_per_element_);

        const void* norm_vec = data_point;
        size_t dim = *((size_t*)this->dist_func_param_);
        std::vector<float> norm_buf(dim);
        if (this->normalize_) {
            float len = this->normalize_vector((float*)data_point, norm_buf.data(), dim);
            memcpy(this->length_memory_ + cur_c * sizeof(float), &len, sizeof(float));
            norm_vec = norm_buf.data();
        }
        memcpy(this->getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(this->getDataByInternalId(cur_c), norm_vec, this->data_size_);

        if (curlevel) {
            this->linkLists_[cur_c] = (char*)malloc(this->size_links_per_element_ * curlevel + 1);
            if (!this->linkLists_[cur_c])
                throw std::runtime_error("OOM: adaptive linklist");
            memset(this->linkLists_[cur_c], 0, this->size_links_per_element_ * curlevel + 1);
        }

        std::vector<dist_t> sample_dists;
        sample_dists.reserve(cfg.sample_k + 1);
        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = this->fstdistfunc_(
                    norm_vec, this->getDataByInternalId(currObj),
                    this->dist_func_param_);
                if (sample_dists.size() < cfg.sample_k + 1) {
                    sample_dists.push_back(curdist);
                }
                for (int lev = maxlevelcopy; lev > curlevel; lev--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        std::unique_lock<std::mutex> lk(this->link_list_locks_[currObj]);
                        unsigned int* d = this->get_linklist(currObj, lev);
                        int sz = this->getListCount(d);
                        auto* dl = (tableint*)(d + 1);
                        for (int i = 0; i < sz; i++) {
                            dist_t dd = this->fstdistfunc_(norm_vec, this->getDataByInternalId(dl[i]), this->dist_func_param_);
                            if (sample_dists.size() < cfg.sample_k + 1) {
                                sample_dists.push_back(dd);
                            }
                            if (dd < curdist) {
                                curdist = dd;
                                currObj = dl[i];
                                changed = true;
                            }
                        }
                    }
                }
            }
            if (!sample_dists.empty()) {
                state = buildState(sample_dists);
                float mean_sample = 0.0f;
                for (auto d : sample_dists)
                    mean_sample += static_cast<float>(d);
                mean_sample /= sample_dists.size();
                if (!ema_initialized_) {
                    ema_mean_dist_ = mean_sample;
                    if (this->cur_element_count >= cfg.ema_warmup)
                        ema_initialized_ = true;
                } else {
                    ema_mean_dist_ = (1.0f - cfg.ema_alpha) * ema_mean_dist_ + cfg.ema_alpha * mean_sample;
                }
            }

            bool epDeleted = this->isMarkedDeleted(enterpoint_copy);

            size_t ef_upper = static_cast<size_t>(std::max(1.0f, static_cast<float>(this->ef_construction_) * 
                                                    state.efConstructionScale(cfg)));
            size_t ef_layer0 = static_cast<size_t>(std::max(1.0f, static_cast<float>(this->ef_construction_) *
                                                    state.efConstructionScaleLayer0(cfg)));

            for (int lev = std::min(curlevel, maxlevelcopy); lev >= 0; lev--) {
                size_t ef_this_lev = (lev == 0) ? ef_layer0 : ef_upper;

                size_t saved_ef = this->ef_construction_;
                this->ef_construction_ = ef_this_lev;
                auto top_candidates = this->searchBaseLayer(currObj, norm_vec, lev);
                this->ef_construction_ = saved_ef;

                if (lev == 0 && dim <= cfg.angular_max_dim) {
                    float thresh = state.angularThreshold(cfg);
                    top_candidates = angularDiversify(norm_vec, std::move(top_candidates), thresh);
                }

                if (epDeleted) {
                    top_candidates.emplace(
                        this->fstdistfunc_(
                            norm_vec,
                            this->getDataByInternalId(enterpoint_copy),
                            this->dist_func_param_),
                        enterpoint_copy);
                    if (top_candidates.size() > this->ef_construction_)
                        top_candidates.pop();
                }
                currObj = this->mutuallyConnectNewElement(norm_vec, cur_c, top_candidates, lev, false);
            }
        } else {
            this->enterpoint_node_ = 0;
            this->maxlevel_ = curlevel;
            this->markElementToPersist(cur_c);
        }

        if (curlevel > maxlevelcopy) {
            this->enterpoint_node_ = cur_c;
            this->maxlevel_ = curlevel;
        }
    }

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void* query_data, size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr) const override {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (this->cur_element_count == 0) return result;

        tableint currObj = this->enterpoint_node_;
        dist_t curdist = this->fstdistfunc_(query_data, this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_);

        for (int level = this->maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int* d =
                    (unsigned int*)this->get_linklist(currObj, level);
                int sz = this->getListCount(d);
                auto* dl = (tableint*)(d + 1);
                for (int i = 0; i < sz; i++) {
                    dist_t dd = this->fstdistfunc_(query_data, this->getDataByInternalId(dl[i]), this->dist_func_param_);
                    if (dd < curdist) {
                        curdist = dd;
                        currObj = dl[i];
                        changed = true;
                    }
                }
            }
        }

        std::vector<dist_t> seed_dist;
        seed_dist.reserve(cfg.sample_k + 1);
        seed_dist.push_back(curdist);
        {
            unsigned int* d = (unsigned int*)this->get_linklist0(currObj);
            int sz = this->getListCount(d);
            auto* dl = (tableint*)(d + 1);
            for (int i = 0; i < sz && seed_dist.size() <= cfg.sample_k; i++) {
                seed_dist.push_back(this->fstdistfunc_(query_data, this->getDataByInternalId(dl[i]), this->dist_func_param_));
            }
        }

        AdaptiveState state = buildState(seed_dist);
        float ef_scale = state.efSearchScale(cfg);
        size_t base_ef = static_cast<size_t>(std::max(this->ef_, k));
        size_t adapted_ef = static_cast<size_t>(std::max(static_cast<float>(k), static_cast<float>(base_ef) * ef_scale));

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> top_candidates;
        if (this->num_deleted_)
            top_candidates = this->template searchBaseLayerST<true, true>(currObj, query_data, adapted_ef, isIdAllowed);
        else
            top_candidates = this->template searchBaseLayerST<false, true>(currObj, query_data, adapted_ef, isIdAllowed);

        while (top_candidates.size() > k) top_candidates.pop();
        while (!top_candidates.empty()) {
            auto& top = top_candidates.top();
            result.push({top.first, this->getExternalLabel(top.second)});
            top_candidates.pop();
        }
        return result;
    }

   private:
    float ema_mean_dist_;
    bool ema_initialized_;

    bool needsConstructionProbe() const {
        return cfg.ml_scale_max > 1.0f || cfg.ef_c_density_gain > 0.0f ||
               cfg.ef_c_variance_gain > 0.0f || cfg.ef_c_skewness_gain > 0.0f ||
               cfg.angular_max_dim > 0;
    }
    AdaptiveState buildState(const std::vector<dist_t>& samples) const {
        AdaptiveState state;
        if (samples.empty()) return state;

        size_t dim = *((size_t*)this->dist_func_param_);

        float expected = ema_initialized_ ? ema_mean_dist_ : static_cast<float>(dim) / 6.0f;
        expected = std::max(expected, 1e-6f);

        float mean = 0.0f;
        for (auto d : samples) mean += static_cast<float>(d);
        mean /= static_cast<float>(samples.size());

        float dense_ref = expected * 0.65f;
        float sparse_ref = expected * 1.35f;
        float t_density = (mean - dense_ref) / (sparse_ref - dense_ref + 1e-10f);
        state.density_score = 1.0f - std::max(0.0f, std::min(1.0f, t_density));

        float var = 0.0f;
        if (samples.size() >= 2) {
            for (auto d : samples) {
                float diff = static_cast<float>(d) - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(samples.size() - 1);
        }
        float var_ref = expected * expected;
        state.variance_score = std::max(0.0f, std::min(1.0f, var / (var_ref + 1e-10f)));

        float sigma = std::sqrt(var + 1e-10f);
        float skew_raw = 0.0f;
        if (samples.size() >= 3) {
            for (auto d : samples) {
                float z = (static_cast<float>(d) - mean) / sigma;
                skew_raw += z * z * z;
            }
            skew_raw /= static_cast<float>(samples.size());
        }
        state.skewness_score = std::tanh(std::abs(skew_raw) / 2.0f);

        float w_sum = cfg.w_density + cfg.w_variance + cfg.w_skewness;
        if (w_sum < 1e-6f) w_sum = 1.0f;
        state.difficulty = (cfg.w_density * state.density_score +
                            cfg.w_variance * state.variance_score +
                            cfg.w_skewness * state.skewness_score) / w_sum;
        state.difficulty = std::max(0.0f, std::min(1.0f, state.difficulty));

        state.is_frontier = (state.density_score < cfg.frontier_density_max) &&
                            (state.variance_score > cfg.frontier_variance_min);

        state.valid = true;
        return state;
    }

    int sampleAdaptiveLevel(const AdaptiveState& state) {
        double scale = (state.valid && state.is_frontier) ? 1.0 : static_cast<double>(state.revSizeScale(cfg));
        double adapted_revSize = this->revSize_ * scale;
        int level = this->getRandomLevel(adapted_revSize);

        bool bootstrapping = (this->cur_element_count < cfg.level_cap_bootstrap);
        int cap = bootstrapping ? std::max(0, this->maxlevel_ + 1) : std::max(0, this->maxlevel_);
        return std::min(level, cap);
    }

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    angularDiversify(
        const void* query,
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidates,
        float threshold) const {
        size_t dim = *((size_t*)this->dist_func_param_);
        const float* q = (const float*)query;

        std::vector<std::pair<dist_t, tableint>> sorted;
        while (!candidates.empty()) {
            sorted.push_back(candidates.top());
            candidates.pop();
        }
        std::reverse(sorted.begin(), sorted.end());
        const size_t angular_dim = std::min(dim, size_t(32));
        auto get_dir = [&](tableint id, std::vector<float>& buf) {
            const float* c = (const float*)this->getDataByInternalId(id);
            float sq = 0.0f;
            for (size_t i = 0; i < angular_dim; i++) {
                buf[i] = q[i] - c[i];
                sq += buf[i] * buf[i];
            }
            float inv = 1.0f / (std::sqrt(sq) + 1e-10f);
            for (size_t i = 0; i < angular_dim; i++) buf[i] *= inv;
        };

        std::vector<std::pair<dist_t, tableint>> accepted, rejected;
        accepted.reserve(this->M_);

        std::vector<std::vector<float>> accepted_dirs;
        accepted_dirs.reserve(this->M_);
        std::vector<float> dir_cand(angular_dim);

        const size_t max_cmp = 4;

        for (auto& cand : sorted) {
            get_dir(cand.second, dir_cand);

            bool too_similar = false;
            size_t n_acc = accepted_dirs.size();
            size_t start = n_acc > max_cmp ? n_acc - max_cmp : 0;
            for (size_t ai = start; ai < n_acc; ai++) {
                float dot = 0.0f;
                for (size_t i = 0; i < angular_dim; i++)
                    dot += dir_cand[i] * accepted_dirs[ai][i];
                if (dot > threshold) {
                    too_similar = true;
                    break;
                }
            }

            if (!too_similar) {
                accepted.push_back(cand);
                accepted_dirs.push_back(dir_cand);
            } else {
                rejected.push_back(cand);
            }
        }

        size_t target = std::min(this->M_, sorted.size());
        for (size_t i = 0; accepted.size() < target && i < rejected.size(); i++)
            accepted.push_back(rejected[i]);

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            result;
        for (auto& p : accepted) result.push(p);
        return result;
    }
};

}