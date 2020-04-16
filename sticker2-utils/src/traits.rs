use anyhow::Result;
use clap::{App, AppSettings, ArgMatches};

pub static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub trait StickerApp
where
    Self: Sized,
{
    fn app() -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Result<Self>;

    fn run(&self) -> Result<()>;
}
