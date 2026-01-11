use puffin::profile_scope;

pub fn format_number<T: ToString>(value: T) -> String {
    let s = value.to_string();
    let mut chars: Vec<char> = s.chars().rev().collect();
    let mut formatted = String::new();

    for (i, c) in chars.iter().enumerate() {
        if i != 0 && i % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(*c);
    }

    formatted.chars().rev().collect()
}