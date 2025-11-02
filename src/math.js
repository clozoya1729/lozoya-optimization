function random_float(min, max, figures = 2) {
    return parseFloat((Math.random() * (max - min) + min).toFixed(figures));
}

function make_random_float_seeded(seed) {
    if (seed === undefined) {
        return random_normal;
    }
    const rng = seeded_random(seed);
    return function (min = 0, max = 1) {
        return min + (max - min) * rng();
    };
}

function seeded_random(seed) {
    // Simple seedable PRNG (Mulberry32)
    let t = seed + 0x6D2B79F5;
    return function () {
        t += 0x6D2B79F5;
        let x = Math.imul(t ^ (t >>> 15), t | 1);
        x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
        return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    }
}

function make_random_normal_seeded(seed) {
    if (seed === undefined) {
        return random_normal
    }
    const rng = seeded_random(seed);
    let spare;
    let hasSpare = false;
    return function () {
        if (hasSpare) {
            hasSpare = false;
            return spare;
        }
        let u, v, s;
        do {
            u = rng() * 2 - 1;
            v = rng() * 2 - 1;
            s = u * u + v * v;
        } while (s === 0 || s >= 1);
        const mul = Math.sqrt(-2.0 * Math.log(s) / s);
        spare = v * mul;
        hasSpare = true;
        return u * mul;
    }
}

function random_normal() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function random_gauss(mean = 0, stdDev = 1) {
    let u1 = Math.random();
    let u2 = Math.random();
    let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * stdDev + mean;
}

function random_float3(min, max, figures = 2) {
    return [
        random_float(min, max, figures),
        random_float(min, max, figures),
        random_float(min, max, figures),
    ]
}

function random_from_array(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}
