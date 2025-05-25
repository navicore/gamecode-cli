use flag_rs::Command;

mod completion;
mod models;
mod prompts;
mod sessions;

pub fn register_commands(root: &mut Command) {
    // Each subcommand module registers itself
    completion::register(root);
    models::register(root);
    prompts::register(root);
    sessions::register(root);
}