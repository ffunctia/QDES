#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector> 
#include <cmath>
#include <algorithm> 
#include <random>
#include <omp.h>

using pybind11::array_t;
using std::complex;
using std::vector;
using std::sqrt;
using std::norm;
using std::conj;
using std::log2;
using std::abs;
using std::min;
using std::pair;

inline int logical_to_state_bit(int qubit_count, int logical_idx)
{
    return qubit_count - 1 - logical_idx;
}

void apply_gate(
    array_t<complex<double>> state_vector,
    array_t<complex<double>> gate,
    array_t<int64_t> target_qubits){

    auto sv  = state_vector.mutable_unchecked<1>();
    auto G   = gate.unchecked<2>();
    auto tq  = target_qubits.unchecked<1>();

    int n = (int)log2(state_vector.size());
    int k = (int)target_qubits.size();
    int n_rest = n - k;
    int dim = 1 << k;

    vector<int> other; 
    other.reserve(n_rest);
    for (int q = 0; q < n; q++){
        bool found = false;
        for (int i = 0; i < k; i++) {
            if (logical_to_state_bit(n, (int)tq(i)) == q) {
                found = true;
                break;
            }
        }
        if (!found) other.push_back(q);
    } 
    #pragma omp parallel
    {
        vector<int> indices(dim);
        vector<complex<double>> col_vec(dim);

        #pragma omp for
        for (int col = 0; col < (1 << n_rest); col++) {
            
            for (int row = 0; row < dim; row++) {
                int idx = 0;
                for (int bp = 0; bp < k; bp++)
                    idx |= ((row >> (k - 1 - bp)) & 1)
                        << logical_to_state_bit(n, (int)tq(bp));
                for (int bp = 0; bp < n_rest; bp++)
                    idx |= ((col >> (n_rest - 1 - bp)) & 1) << other[bp];
                indices[row] = idx;
            }

            for (int i = 0; i < dim; i++)
                col_vec[i] = sv(indices[i]);

            for (int i = 0; i < dim; i++) {
                complex<double> s = 0;
                for (int j = 0; j < dim; j++)
                    s += G(i, j) * col_vec[j];
                sv(indices[i]) = s;
            }
        }
    }
}

/*
void apply_gate(
    array_t<complex<double>> state_vector,
    array_t<complex<double>> gate,
    array_t<int64_t> target_qubits){

    auto sv  = state_vector.mutable_unchecked<1>();
    auto G   = gate.unchecked<2>();
    auto tq  = target_qubits.unchecked<1>();

    int n = (int)log2(state_vector.size());
    int k = (int)target_qubits.size();
    int n_rest = n - k;

    vector<int> other; 
    for (int q = 0; q < n; q++){
        bool found = false;
        for (int i = 0; i < k; i++) {
            if (logical_to_state_bit(n, (int)tq(i)) == q) {
                found = true;
                break;
            }
        }
        if (!found) other.push_back(q);
    } 

    #pragma omp parallel for
    for (int col = 0; col < (1 << n_rest); col++) {
        vector<int> indices(1 << k);
        for (int row = 0; row < (1 << k); row++) {
            int idx = 0;
            for (int bp = 0; bp < k; bp++)
                idx |= ((row >> (k - 1 - bp)) & 1)
                    << logical_to_state_bit(n, (int)tq(bp));
            for (int bp = 0; bp < n_rest; bp++)
                idx |= ((col >> (n_rest - 1 - bp)) & 1) << other[bp];
            indices[row] = idx;
        }

        vector<complex<double>> col_vec(1 << k);
        for (int i = 0; i < (1 << k); i++)
            col_vec[i] = sv(indices[i]);

        for (int i = 0; i < (1 << k); i++) {
            complex<double> s = 0;
            for (int j = 0; j < (1 << k); j++)
                s += G(i, j) * col_vec[j];
            sv(indices[i]) = s;
        }
    }
} 
*/

array_t<complex<double>> kron_merge(
    array_t<complex<double>> a,
    array_t<complex<double>> b)
{
    const auto* A = a.data();
    const auto* B = b.data();

    int64_t na = a.size();
    int64_t nb = b.size();

    array_t<complex<double>> out(na * nb);
    auto* O = out.mutable_data();

    #pragma omp parallel for
    for (int64_t i = 0; i < na; i++) {
        complex<double> ai = A[i];
        int64_t base = i * nb;

        for (int64_t j = 0; j < nb; j++) {
            O[base + j] = ai * B[j];
        }
    }

    return out;
}

bool is_entangled_no_copy(
    array_t<complex<double>> state_vector,
    array_t<int64_t> subset,
    array_t<int64_t> complement,
    double EPS,
    double TOL,
    int64_t CHUNK)
{
    auto SV = state_vector.unchecked<1>();
    auto sub = subset.unchecked<1>();
    auto comp = complement.unchecked<1>();
    
    int k = (int)subset.size();
    int n_rest = (int)complement.size();
    int64_t size_a = 1ULL << k;
    int64_t size_b = 1ULL << n_rest;

    vector<int> sub_bits(k), comp_bits(n_rest);
    for (int bp = 0; bp < k; bp++)
        sub_bits[bp] = logical_to_state_bit(k + n_rest, (int)sub(bp));
    for (int bp = 0; bp < n_rest; bp++)
        comp_bits[bp] = logical_to_state_bit(k + n_rest, (int)comp(bp));

    vector<complex<double>> ref(size_a);
    int64_t ref_col = -1;
    double ref_norm = 0.0;

    for (int64_t j = 0; j < size_b; j++){

        int64_t idx_j = 0;
        for (int bp = 0; bp < n_rest; bp++) {
            if ((j >> (n_rest - 1 - bp)) & 1) idx_j |= (1ULL << comp_bits[bp]);
        }

        double norm2 = 0.0;
        for (int64_t i = 0; i < size_a; i++) {
            int64_t idx_i = 0;
            for (int bp = 0; bp < k; bp++) {
                if ((i >> (k - 1 - bp)) & 1) idx_i |= (1ULL << sub_bits[bp]);
            }
            norm2 += norm(SV(idx_j | idx_i));
        }

        double norm = sqrt(norm2);
        if (norm > EPS) {
            ref_col = j;
            ref_norm = norm;
            for (int64_t i = 0; i < size_a; i++) {
                int64_t idx_i = 0;
                for (int bp = 0; bp < k; bp++) {
                    if ((i >> (k - 1 - bp)) & 1) idx_i |= (1ULL << sub_bits[bp]);
                }
                ref[i] = SV(idx_j | idx_i) / norm;
            }
            break;
        }

    }

    if (ref_col < 0) return false;

    std::atomic<bool> entangled_flag{false};

    #pragma omp parallel for schedule(dynamic, CHUNK)
    for (int64_t j = 0; j < size_b; j++) {
        if (entangled_flag.load(std::memory_order_relaxed)) continue;
        if (j == ref_col) continue;

        int64_t idx_j = 0;
        for (int bp = 0; bp < n_rest; bp++) {
            if ((j >> (n_rest - 1 - bp)) & 1) idx_j |= (1ULL << comp_bits[bp]);
        }

        double norm2 = 0.0;
        for (int64_t i = 0; i < size_a; i++) {
            int64_t idx_i = 0;
            for (int bp = 0; bp < k; bp++) {
                if ((i >> (k - 1 - bp)) & 1) idx_i |= (1ULL << sub_bits[bp]);
            }
            norm2 += norm(SV(idx_j | idx_i));
        }
        double norm = sqrt(norm2);
        if (norm <= EPS) continue;

        complex<double> ip = 0;
        for (int64_t i = 0; i < size_a; i++) {
            int64_t idx_i = 0;
            for (int bp = 0; bp < k; bp++) {
                if ((i >> (k - 1 - bp)) & 1) idx_i |= (1ULL << sub_bits[bp]);
            }
            ip += conj(ref[i]) * SV(idx_j | idx_i);
        }
        double ip_mag = abs(ip) / norm;

        if (ip_mag < 1.0 - TOL) {
            entangled_flag.store(true, std::memory_order_relaxed);
        }
    }
    return entangled_flag.load();
}

pair<array_t<complex<double>>, array_t<complex<double>>> factor_rank1_no_copy(
    array_t<complex<double>> state_vector,
    array_t<int64_t> subset,
    array_t<int64_t> complement,
    double EPS)
{
    auto SV = state_vector.unchecked<1>();
    auto sub = subset.unchecked<1>();
    auto comp = complement.unchecked<1>();
    
    int k = (int)subset.size();
    int n_rest = (int)complement.size();
    int64_t size_a = 1ULL << k;
    int64_t size_b = 1ULL << n_rest;

    vector<int> sub_bits(k), comp_bits(n_rest);
    for (int bp = 0; bp < k; bp++)
        sub_bits[bp] = logical_to_state_bit(k + n_rest, (int)sub(bp));
    for (int bp = 0; bp < n_rest; bp++)
        comp_bits[bp] = logical_to_state_bit(k + n_rest, (int)comp(bp));

    array_t<complex<double>> sv_a(size_a);
    array_t<complex<double>> sv_b(size_b);
    auto Ua = sv_a.mutable_unchecked<1>();
    auto Vb = sv_b.mutable_unchecked<1>();

    int64_t ref_col = -1;
    double ref_norm = 0.0;

    for (int64_t j = 0; j < size_b; j++) {
        int64_t idx_j = 0;
        for (int bp = 0; bp < n_rest; bp++) {
            if ((j >> (n_rest - 1 - bp)) & 1) idx_j |= (1ULL << comp_bits[bp]);
        }
        double norm2 = 0.0;
        for (int64_t i = 0; i < size_a; i++) {
            int64_t idx_i = 0;
            for (int bp = 0; bp < k; bp++) {
                if ((i >> (k - 1 - bp)) & 1) idx_i |= (1ULL << sub_bits[bp]);
            }
            norm2 += norm(SV(idx_j | idx_i));
        }
        double norm = sqrt(norm2);
        if (norm > EPS) { ref_col = j; ref_norm = norm; break; }
    }

    if (ref_col < 0) {
        for (int64_t i = 0; i < size_a; i++) Ua(i) = 0;
        for (int64_t j = 0; j < size_b; j++) Vb(j) = 0;
        return {sv_a, sv_b};
    }

    int64_t ref_idx_j = 0;
    for (int bp = 0; bp < n_rest; bp++) {
        if ((ref_col >> (n_rest - 1 - bp)) & 1) ref_idx_j |= (1ULL << comp_bits[bp]);
    }

    #pragma omp parallel for
    for (int64_t i = 0; i < size_a; i++) {
        int64_t idx_i = 0;
        for (int bp = 0; bp < k; bp++) {
            if ((i >> (k - 1 - bp)) & 1) idx_i |= (1ULL << sub_bits[bp]);
        }
        Ua(i) = SV(ref_idx_j | idx_i) / ref_norm;
    }

    #pragma omp parallel for
    for (int64_t j = 0; j < size_b; j++) {
        int64_t idx_j = 0;
        for (int bp = 0; bp < n_rest; bp++) {
            if ((j >> (n_rest - 1 - bp)) & 1) idx_j |= (1ULL << comp_bits[bp]);
        }
        complex<double> ip = 0;
        for (int64_t i = 0; i < size_a; i++) {
            int64_t idx_i = 0;
            for (int bp = 0; bp < k; bp++) {
                if ((i >> (k - 1 - bp)) & 1) idx_i |= (1ULL << sub_bits[bp]);
            }
            ip += conj(Ua(i)) * SV(idx_j | idx_i);
        }
        Vb(j) = ip;
    }

    double vb_norm2 = 0.0;
    for (int64_t j = 0; j < size_b; j++) vb_norm2 += norm(Vb(j));
    double vb_norm = sqrt(vb_norm2);
    if (vb_norm > EPS) {
        for (int64_t j = 0; j < size_b; j++) Vb(j) /= vb_norm;
        for (int64_t i = 0; i < size_a; i++) Ua(i) *= vb_norm;
    }

    return {sv_a, sv_b};
}

double measure_probability(
    array_t<complex<double>> state_vector,
    int64_t local_idx)
{
    auto SV = state_vector.unchecked<1>();
    int64_t n = state_vector.size();
    int k = (int)log2((double)n);
    int shift = k - 1 - (int)local_idx;

    double p1 = 0.0;
    #pragma omp parallel for reduction(+:p1)
    for (int64_t i = 0; i < n; i++) {
        if ((i >> shift) & 1) {
            p1 += norm(SV(i));
        }
    }
    return p1;
}

void collapse_and_normalize(
    array_t<complex<double>> state_vector,
    int64_t local_idx,
    int result,
    double norm_value,
    double norm_epsilon)
{
    auto sv = state_vector.mutable_unchecked<1>();
    int64_t n = state_vector.size();
    int k = (int)log2((double)n);
    int shift = k - 1 - (int)local_idx;
    double inv_norm = (norm_value > norm_epsilon) ? (1.0 / sqrt(norm_value)) : 0.0;

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        int bit = (int)((i >> shift) & 1);
        if (bit != result) {
            sv(i) = 0.0;
        } else {
            sv(i) *= inv_norm;
        }
    }
}

array_t<complex<double>> collapse_and_extract(
    array_t<complex<double>> state_vector,
    int64_t local_idx,
    int result,
    double norm_value,
    double norm_epsilon)
{
    auto SV = state_vector.unchecked<1>();
    int64_t n = state_vector.size();
    int k = (int)log2((double)n);
    int shift = k - 1 - (int)local_idx;
    int64_t n_out = n / 2;
    double inv_norm = (norm_value > norm_epsilon) ? (1.0 / sqrt(norm_value)) : 0.0;

    array_t<complex<double>> out(n_out);
    auto O = out.mutable_unchecked<1>();

    int64_t low_mask = (shift > 0) ? ((1LL << shift) - 1) : 0;

    #pragma omp parallel for
    for (int64_t o = 0; o < n_out; o++) {
        int64_t low = o & low_mask;
        int64_t high = (o >> shift) << (shift + 1);
        int64_t full_idx = high | (((int64_t)result) << shift) | low;
        O(o) = SV(full_idx) * inv_norm;
    }
    return out;
}

// sampling

array_t<int64_t> sample_group_shots(
    array_t<complex<double>> state_vector,
    array_t<int64_t> target_shifts,
    int64_t count,
    uint64_t seed,
    double norm_epsilon)
{
    auto SV = state_vector.unchecked<1>();
    auto shifts = target_shifts.unchecked<1>();

    int64_t n = state_vector.size();
    int n_qubits = (int)log2((double)n);
    int k = (int)target_shifts.size();

    array_t<int64_t> samples(count);
    auto S = samples.mutable_unchecked<1>();

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> rands(count);
    for (int64_t i = 0; i < count; i++) {
        rands[i] = dist(rng);
    }
    std::sort(rands.begin(), rands.end());

    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;
    
    std::vector<double> block_sums(num_threads, 0.0);
    int64_t block_size = (n + num_threads - 1) / num_threads;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = tid * block_size;
        int64_t end = std::min(start + block_size, n);
        double local_sum = 0.0;

        for (int64_t i = start; i < end; i++) {
            local_sum += std::norm(SV(i));
        }
        block_sums[tid] = local_sum;
    }

    std::vector<double> block_cdf(num_threads + 1, 0.0);
    for (int i = 0; i < num_threads; i++) {
        block_cdf[i + 1] = block_cdf[i] + block_sums[i];
    }
    double total_norm = block_cdf[num_threads];

    if (total_norm < norm_epsilon) {
        for (int64_t i = 0; i < count; i++) S(i) = 0;
        return samples;
    }

    for (int64_t i = 0; i < count; i++) {
        rands[i] *= total_norm;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = tid * block_size;
        int64_t end = std::min(start + block_size, n);

        auto it_start = std::lower_bound(rands.begin(), rands.end(), block_cdf[tid]);
        auto it_end = std::lower_bound(rands.begin(), rands.end(), block_cdf[tid + 1]);

        if (it_start != it_end) {
            double current_cdf = block_cdf[tid];
            auto current_rand_it = it_start;

            for (int64_t i = start; i < end; i++) {
                current_cdf += std::norm(SV(i));
                
                while (current_rand_it != it_end && current_cdf >= *current_rand_it) {
                    int64_t outcome = 0;
                    
                    for (int bp = 0; bp < k; bp++) {
                        int shift = logical_to_state_bit(n_qubits, (int)shifts(bp));
                        int64_t bit = (i >> shift) & 1LL;
                        outcome |= (bit << (k - 1 - bp));
                    }
                    
                    int64_t sample_idx = std::distance(rands.begin(), current_rand_it);
                    S(sample_idx) = outcome;
                    ++current_rand_it;
                }
                
                if (current_rand_it == it_end) break;
            }

            while (current_rand_it != it_end) {
                int64_t last_i = end - 1;
                if (last_i < 0) last_i = 0;
                int64_t outcome = 0;
                for (int bp = 0; bp < k; bp++) {
                    int shift = logical_to_state_bit(n_qubits, (int)shifts(bp));
                    int64_t bit = (last_i >> shift) & 1LL;
                    outcome |= (bit << (k - 1 - bp));
                }
                int64_t sample_idx = std::distance(rands.begin(), current_rand_it);
                S(sample_idx) = outcome;
                ++current_rand_it;
            }
        }
    }

    return samples;
}

PYBIND11_MODULE(utils_cside, m) {
    m.def("apply_gate", &apply_gate);
    m.def("kron_merge", &kron_merge);
    m.def("is_entangled_no_copy", &is_entangled_no_copy);
    m.def("factor_rank1_no_copy", &factor_rank1_no_copy);
    m.def("measure_probability", &measure_probability);
    m.def("collapse_and_normalize", &collapse_and_normalize);
    m.def("collapse_and_extract", &collapse_and_extract);
    m.def("sample_group_shots", &sample_group_shots);
}