use flag_rs::Command;

mod completion;
pub mod mcp;
mod models;
mod prompts;
mod sessions;

pub fn register_commands(root: &mut Command) {
    // Each subcommand module registers itself
    completion::register(root);
    mcp::register(root);
    models::register(root);
    prompts::register(root);
    sessions::register(root);
}