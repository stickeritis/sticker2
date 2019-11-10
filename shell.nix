with import <nixpkgs> {};

stdenv.mkDerivation rec {
  name = "sticker-env";
  env = buildEnv { name = name; paths = buildInputs; };

  nativeBuildInputs = [
    latest.rustChannels.stable.rust
    pkgconfig
  ];

  buildInputs = [
    curl
    openssl
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;
}
