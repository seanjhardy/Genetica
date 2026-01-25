import package::types::Point;
import package::constants::M_PI;

// Computes the signed angle from vector u to v in radians (range [-pi, pi]).
fn signed_angle(u: vec2<f32>, v: vec2<f32>) -> f32 {
    // For 2D: atan2(cross(u,v), dot(u,v))
    let cross_z = u.x * v.y - u.y * v.x;
    let d = dot(u, v);
    return atan2(cross_z, d);
}

fn rotate2d(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}


fn unsigned_angle(u: vec2<f32>, v: vec2<f32>) -> f32 {
    let eps = 1e-8;
    let lu = length(u);
    let lv = length(v);
    if (lu < eps || lv < eps) {
        return 0.0;
    }
    let c = clamp(dot(u, v) / (lu * lv), -1.0, 1.0);
    return acos(c); // [0, pi]
}

fn distance_constraint(
    a: Point,
    b: Point,
    rest_length: f32,
    strength: f32
) -> vec4<f32> {
    let pos_a = a.pos;
    let pos_b = b.pos;

    // Your mass model
    let a_mass = a.radius * a.radius;
    let b_mass = b.radius * b.radius;

    let delta = pos_b - pos_a;
    let dist = length(delta);

    let eps = 1e-8;
    if (dist < eps) {
        // No well-defined direction to separate; do nothing (or pick a fallback dir)
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Inverse masses (mass == 0 => fixed)
    let invA = select(0.0, 1.0 / a_mass, a_mass > 0.0);
    let invB = select(0.0, 1.0 / b_mass, b_mass > 0.0);
    let invSum = invA + invB;

    if (invSum <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let diff = (dist - rest_length) / dist; // scalar
    let s = clamp(strength, 0.0, 1.0);

    // Shared correction along delta (same as n * (dist-rest))
    let correction = delta * (s * diff / invSum);

    // Return per-point position deltas
    let dA =  correction * invA;
    let dB = -correction * invB;

    return vec4<f32>(dA, dB);
}

// Rotates A and B about C to move angle ABC toward `theta`.
// - A, B, C: point positions (2D)
// - massA, massB: masses for A and B (>= 0; if 0, that point won't move)
// - strength: [0..1] how much of the correction to apply this call
// - theta: desired signed angle at C from CA to CB (radians), in [-pi, pi]
//
// Returns updated (A, B). C is unchanged.
fn rotate_points_to_angle(
    A: Point,
    B: Point,
    anchor: vec2<f32>,
    strength: f32,
    theta: f32,
) -> vec4<f32> {
    // Vectors from C
    var u = A.pos - anchor; // CA
    var v = B.pos - anchor; // CB

    // Guard against degenerate vectors
    let eps = 1e-8;
    let lu = length(u);
    let lv = length(v);
    if (lu < eps || lv < eps) {
        // Can't define an angle; return unchanged
        return vec4<f32>(A.pos, B.pos);
    }

    // Current signed angle at C
    let cur = signed_angle(u, v);

    // Smallest signed delta to reach theta from cur, wrapped to [-pi, pi]
    var delta = theta - cur;
    // wrap
    let pi = M_PI;
    let two_pi = M_PI * 2.0;
    if (delta >  pi) { delta = delta - two_pi; }
    if (delta < -pi) { delta = delta + two_pi; }

    // Apply only a fraction (strength) of the correction
    let corr = delta * clamp(strength, 0.0, 1.0);

    // Mass-weighted split: heavier moves less.
    // If mass==0, treat as "immovable"? Here: 0 mass => infinite inverse mass => moves most.
    // If you prefer 0 mass immovable, swap the invMass logic accordingly.
    let massA = A.radius * A.radius;
    let massB = B.radius * B.radius;
    let invA = select(0.0, 1.0 / massA, massA > 0.0);
    let invB = select(0.0, 1.0 / massB, massB > 0.0);
    let invSum = invA + invB;

    // If both immovable, do nothing
    if (invSum <= 0.0) {
        return vec4<f32>(A.pos, B.pos);
    }

    // We rotate A by -corr * wA and B by +corr * wB so the relative angle changes by corr.
    let wA = invA / invSum;
    let wB = invB / invSum;

    let u2 = rotate2d(u, -corr * wA);
    let v2 = rotate2d(v,  corr * wB);

    let A2 = anchor + u2;
    let B2 = anchor + v2;

    return vec4<f32>(A2, B2);
}

// Rotates A and B about C to change the signed angle at C by `corr` radians.
// Mass-weighted: heavier points move less.
// mass==0 => fixed (invMass=0).
fn apply_angle_correction(
    A: Point,
    B: Point,
    anchor: vec2<f32>,
    corr: f32,
) -> vec4<f32> {
    let u = A.pos - anchor;
    let v = B.pos - anchor;

    let eps = 1e-8;
    if (length(u) < eps || length(v) < eps) {
        return vec4<f32>(A.pos, B.pos);
    }

    let massA = A.radius * A.radius;
    let massB = B.radius * B.radius;
    let invA = select(0.0, 1.0 / massA, massA > 0.0);
    let invB = select(0.0, 1.0 / massB, massB > 0.0);
    let invSum = invA + invB;

    if (invSum <= 0.0) {
        return vec4<f32>(A.pos, B.pos);
    }

    let wA = invA / invSum;
    let wB = invB / invSum;

    // Rotate A opposite direction, B same direction; relative angle changes by corr.
    let u2 = rotate2d(u, -corr * wA);
    let v2 = rotate2d(v,  corr * wB);

    return vec4<f32>(anchor + u2, anchor + v2);
}

// Enforce that the *smallest* angle at C (in [0, pi]) is >= minAngle, exactly.
// If violated, applies the full correction in one shot (no strength parameter).
//
// Returns updated (A, B) packed in vec4(A.xy, B.xy).
fn min_angle_constraint(
    A: Point,
    B: Point,
    anchor: vec2<f32>,
    minAngle: f32, // radians, expected in [0, pi]
) -> vec4<f32> {
    let minClamped = clamp(minAngle, 0.0, M_PI);

    let u = A.pos - anchor;
    let v = B.pos - anchor;

    // Current smallest (unsigned) angle in [0, pi]
    let curSmall = unsigned_angle(u, v);

    // Already satisfies
    if (curSmall >= minClamped) {
        return vec4<f32>(A.pos, B.pos);
    }

    // We want to increase the smallest angle by exactly (minClamped - curSmall),
    // while keeping the current bend direction.
    let s = signed_angle(u, v);

    // If the angle is near 0, signed angle can be 0; choose + direction by default.
    let signDir = select(1.0, -1.0, s < 0.0);

    // Desired signed target is +/- minClamped on the same side as current bend.
    let targetSigned = signDir * minClamped;

    // Current signed angle is s (in [-pi, pi]). Compute exact correction to reach target.
    // Wrap delta into [-pi, pi] to ensure shortest path (though here target is on same side).
    var delta = targetSigned - s;
    if (delta >  M_PI) { delta -= M_PI * 2.0; }
    if (delta < -M_PI) { delta += M_PI * 2.0; }

    // Apply full correction (no strength scaling)
    return apply_angle_correction(A, B, anchor, delta);
}
