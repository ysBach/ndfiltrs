use std::{error::Error, hint::black_box};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndfiltrs::benchmark::{
    generic_2d, specialized_2d, verify_candidate_f32, verify_candidate_f64, BenchmarkPolicy,
};

type FilterF32 = fn(
    &[f32],
    [usize; 2],
    [usize; 2],
    [&str; 2],
    f32,
    BenchmarkPolicy,
) -> Result<Vec<f32>, Box<dyn Error>>;

type FilterF64 = fn(
    &[f64],
    [usize; 2],
    [usize; 2],
    [&str; 2],
    f64,
    BenchmarkPolicy,
) -> Result<Vec<f64>, Box<dyn Error>>;

struct F32Case {
    name: &'static str,
    input: Vec<f32>,
    shape: [usize; 2],
    size: [usize; 2],
    mode: [&'static str; 2],
    cval: f32,
    policy: BenchmarkPolicy,
}

struct F64Case {
    name: &'static str,
    input: Vec<f64>,
    shape: [usize; 2],
    size: [usize; 2],
    mode: [&'static str; 2],
    cval: f64,
    policy: BenchmarkPolicy,
}

fn benchmark_f32_case(criterion: &mut Criterion, case: &F32Case) {
    verify_candidate_f32(
        &case.input,
        case.shape,
        case.size,
        case.mode,
        case.cval,
        case.policy,
    )
    .unwrap_or_else(|error| {
        panic!(
            "benchmark parity preflight failed for {}: {error}",
            case.name
        )
    });

    let mut group = criterion.benchmark_group("filter2d_kernel");
    for (name, implementation) in [
        ("generic_nd_serial", generic_2d::<f32> as FilterF32),
        ("specialized_2d", specialized_2d::<f32> as FilterF32),
    ] {
        group.bench_with_input(BenchmarkId::new(name, case.name), &case.input, |b, data| {
            b.iter(|| {
                implementation(
                    black_box(data),
                    case.shape,
                    case.size,
                    case.mode,
                    case.cval,
                    case.policy,
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

fn benchmark_f64_case(criterion: &mut Criterion, case: &F64Case) {
    verify_candidate_f64(
        &case.input,
        case.shape,
        case.size,
        case.mode,
        case.cval,
        case.policy,
    )
    .unwrap_or_else(|error| {
        panic!(
            "benchmark parity preflight failed for {}: {error}",
            case.name
        )
    });

    let mut group = criterion.benchmark_group("filter2d_kernel");
    for (name, implementation) in [
        ("generic_nd_serial", generic_2d::<f64> as FilterF64),
        ("specialized_2d", specialized_2d::<f64> as FilterF64),
    ] {
        group.bench_with_input(BenchmarkId::new(name, case.name), &case.input, |b, data| {
            b.iter(|| {
                implementation(
                    black_box(data),
                    case.shape,
                    case.size,
                    case.mode,
                    case.cval,
                    case.policy,
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

fn compare_traversals(criterion: &mut Criterion) {
    let f32_cases = [F32Case {
        name: "f32-32x24-3x5-constant-mirror-omit",
        input: (0..32 * 24)
            .map(|index| match index % 19 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                _ => (index % 97) as f32,
            })
            .collect(),
        shape: [32, 24],
        size: [3, 5],
        mode: ["constant", "mirror"],
        cval: -2.0,
        policy: BenchmarkPolicy::Omit {
            ignore_inf: true,
            min_count: 2,
        },
    }];
    let f64_cases = [
        F64Case {
            name: "f64-256x256-5x5-reflect",
            input: (0..256 * 256).map(|index| (index % 97) as f64).collect(),
            shape: [256, 256],
            size: [5, 5],
            mode: ["reflect", "reflect"],
            cval: 0.0,
            policy: BenchmarkPolicy::Propagate,
        },
        F64Case {
            name: "f64-18x11-7x3-nearest-wrap-omit",
            input: (0..18 * 11)
                .map(|index| match index % 23 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    _ => (index % 53) as f64,
                })
                .collect(),
            shape: [18, 11],
            size: [7, 3],
            mode: ["nearest", "wrap"],
            cval: 4.0,
            policy: BenchmarkPolicy::Omit {
                ignore_inf: false,
                min_count: 3,
            },
        },
        F64Case {
            name: "f64-9x13-3x1-mirror-wrap-omit-ignore-inf",
            input: (0..9 * 13)
                .map(|index| match index % 17 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    _ => (index % 31) as f64,
                })
                .collect(),
            shape: [9, 13],
            size: [3, 1],
            mode: ["mirror", "wrap"],
            cval: -1.0,
            policy: BenchmarkPolicy::Omit {
                ignore_inf: true,
                min_count: 1,
            },
        },
    ];

    for case in &f32_cases {
        benchmark_f32_case(criterion, case);
    }
    for case in &f64_cases {
        benchmark_f64_case(criterion, case);
    }
}

criterion_group!(benches, compare_traversals);
criterion_main!(benches);
