
const MATCH_SCORE: i32 = 1;
const MISMATCH_SCORE: i32 = -1;
const GAP_SCORE: i32 = -1;

pub fn gene_similarity_score(a: &Vec<u8>, b: &Vec<u8>) -> f64 {
    let n = a.len();
    let m = b.len();

    let mut dp = vec![vec![0i32; m + 1]; n + 1];
    let mut max_score = 0;

    for i in 1..=n {
        for j in 1..=m {
            let match_or_mismatch = if a[i - 1] == b[j - 1] { MATCH_SCORE } else { MISMATCH_SCORE };
            let score_diag = dp[i - 1][j - 1] + match_or_mismatch;
            let score_up = dp[i - 1][j] + GAP_SCORE;
            let score_left = dp[i][j - 1] + GAP_SCORE;

            dp[i][j] = 0.max(score_diag.max(score_up.max(score_left)));
            max_score = max_score.max(dp[i][j]);
        }
    }

    let max_possible = (n.min(m) as i32) * MATCH_SCORE;
    if max_possible <= 0 { return 0.0; }
    (max_score as f64 / max_possible as f64).clamp(0.0, 1.0)
}