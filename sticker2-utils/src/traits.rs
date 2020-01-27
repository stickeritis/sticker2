use clap::{App, AppSettings, ArgMatches};

pub static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub trait StickerApp {
    fn app() -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Self;

    fn run(&self);
}
