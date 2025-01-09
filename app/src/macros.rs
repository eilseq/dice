#[macro_export]
macro_rules! handle_result {
    ($result:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                post!("Dice Error: {}", err);
                return;
            }
        }
    };
}

#[macro_export]
macro_rules! handle_option {
    ($result:expr, $msg:expr) => {
        match $result {
            Some(value) => value,
            None => {
                post!("Dice Error: {}", $msg);
                return;
            }
        }
    };
}
