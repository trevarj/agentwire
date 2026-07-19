{
  description = "Owner-only IRC bridge for live Codex and OpenCode sessions";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      packageFor = pkgs: pkgs.python3Packages.buildPythonApplication {
        pname = "irc-agent-bridge";
        version = "0.1.0";
        src = ./.;
        pyproject = true;
        build-system = [ pkgs.python3Packages.setuptools ];
        dependencies = [ pkgs.python3Packages.aiohttp ];
        nativeCheckInputs = with pkgs.python3Packages; [ pytest pytest-asyncio ] ++ [ pkgs.ruff ];
        checkPhase = ''
          runHook preCheck
          ruff check src tests
          pytest -q
          runHook postCheck
        '';
      };
      appFor = pkgs: package: command: pkgs.writeShellApplication {
        name = "irc-bridge-${command}";
        runtimeInputs = [ package ];
        text = ''
          config_path="''${IRC_BRIDGE_CONFIG:-$HOME/.config/irc-bridge/config.toml}"
          exec irc-bridge --config "$config_path" ${command} "$@"
        '';
      };
    in {
      packages = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in {
          default = packageFor pkgs;
        });

      checks = forAllSystems (system: {
        default = self.packages.${system}.default;
      });

      apps = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          package = self.packages.${system}.default;
          mk = command: {
            type = "app";
            program = "${appFor pkgs package command}/bin/irc-bridge-${command}";
            meta.description = "Run the IRC agent bridge ${command} command";
          };
        in {
          default = mk "stack";
          stack = mk "stack";
          bridge = mk "run";
          doctor = mk "doctor";
          sync-cert = mk "sync-cert";
          codex-tui = mk "codex-tui";
          opencode-tui = mk "opencode-tui";
        });

      devShells = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in {
          default = pkgs.mkShell {
            packages = [
              (pkgs.python3.withPackages (pythonPackages: with pythonPackages; [
                aiohttp
                pytest
                pytest-asyncio
              ]))
              pkgs.ruff
            ];
          };
        });
    };
}
