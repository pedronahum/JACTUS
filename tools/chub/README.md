# JACTUS — Context Hub Docs

Agent-optimized documentation for [Context Hub](https://github.com/andrewyng/context-hub) (`chub`).

## Use locally (BYOD)

Point your local `chub` at this folder so agents can fetch JACTUS docs without waiting for a PR to merge:

```bash
# 1. Build the registry
chub build /path/to/JACTUS/tools/chub/ -o /tmp/jactus-chub/

# 2. Add it as a local source (~/.chub/config.yaml)
cat >> ~/.chub/config.yaml << 'EOF'
sources:
  - name: jactus
    path: /tmp/jactus-chub
EOF

# 3. Update and verify
chub update
chub search jactus
chub get docs/contracts --lang py
chub get docs/contracts --file references/contract-types.md
chub get docs/contracts --file references/risk-factors.md
chub get docs/contracts --file references/array-mode.md
```

## Files

```
tools/chub/
├── DOC.md                    # main entry — install, quick start, CLI, observers
├── contract-types.md         # parameter examples for all 18 ACTUS contract types
├── risk-factors.md           # observer types and custom observer implementation
├── array-mode.md             # batch/portfolio API, GPU/TPU patterns
└── README.md                 # this file
```

## Contributing to the public registry

These files are also submitted as a PR to [andrewyng/context-hub](https://github.com/andrewyng/context-hub)
under `content/jactus/`. Once merged, any agent running `chub search jactus` will find them automatically.
