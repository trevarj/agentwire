{
  description = "Agentwire: an owner-only IRC bridge for Codex, OpenCode, and Claude sessions";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      packageFor = pkgs: pkgs.python3Packages.buildPythonApplication {
        pname = "agentwire";
        version = "0.1.0";
        src = ./.;
        pyproject = true;
        build-system = [ pkgs.python3Packages.setuptools ];
        dependencies = with pkgs.python3Packages; [ aiohttp claude-agent-sdk ];
        nativeCheckInputs = with pkgs.python3Packages; [ pytest pytest-asyncio ] ++ [ pkgs.ruff ];
        checkPhase = ''
          runHook preCheck
          ruff check src tests
          pytest -q
          runHook postCheck
        '';
      };
      appFor = pkgs: package: command: pkgs.writeShellApplication {
        name = "agentwire-${command}";
        runtimeInputs = [ package ];
        text = ''
          if [ -n "''${AGENTWIRE_CONFIG:-}" ]; then
            config_path="$AGENTWIRE_CONFIG"
          elif [ -f "$HOME/.config/agentwire/config.toml" ]; then
            config_path="$HOME/.config/agentwire/config.toml"
          else
            config_path="''${IRC_BRIDGE_CONFIG:-$HOME/.config/irc-bridge/config.toml}"
          fi
          exec agentwire --config "$config_path" ${command} "$@"
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
            program = "${appFor pkgs package command}/bin/agentwire-${command}";
            meta.description = "Run the Agentwire ${command} command";
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
                claude-agent-sdk
                pytest
                pytest-asyncio
              ]))
              pkgs.ruff
            ];
          };
        });
    };
}
