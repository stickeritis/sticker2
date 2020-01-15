use std::io::stdout;

use clap::{crate_version, App, AppSettings, Arg, Shell, SubCommand};

pub mod progress;

pub mod save;

pub mod sent_proc;

mod subcommands;

pub mod traits;
use traits::StickerApp;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
    AppSettings::SubcommandRequiredElseHelp,
];

fn main() {
    // Known subapplications.
    let apps = vec![
        subcommands::AnnotateApp::app(),
        subcommands::FinetuneApp::app(),
        subcommands::PrepareApp::app(),
    ];

    let cli = App::new("sticker")
        .settings(DEFAULT_CLAP_SETTINGS)
        .about("A neural sequence labeler")
        .version(crate_version!())
        .subcommands(apps)
        .subcommand(
            SubCommand::with_name("completions")
                .about("Generate completion scripts for your shell")
                .setting(AppSettings::ArgRequiredElseHelp)
                .arg(Arg::with_name("shell").possible_values(&Shell::variants())),
        );
    let matches = cli.clone().get_matches();

    match matches.subcommand_name().unwrap() {
        "annotate" => {
            subcommands::AnnotateApp::parse(matches.subcommand_matches("annotate").unwrap()).run()
        }
        "completions" => {
            let shell = matches
                .subcommand_matches("completions")
                .unwrap()
                .value_of("shell")
                .unwrap();
            write_completion_script(cli, shell.parse::<Shell>().unwrap());
        }
        "finetune" => {
            subcommands::FinetuneApp::parse(matches.subcommand_matches("finetune").unwrap()).run()
        }
        "prepare" => {
            subcommands::PrepareApp::parse(matches.subcommand_matches("prepare").unwrap()).run()
        }
        _unknown => unreachable!(),
    }
}

fn write_completion_script(mut cli: App, shell: Shell) {
    cli.gen_completions_to("sticker", shell, &mut stdout());
}
