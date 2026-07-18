# IC device packs

Each YAML file identifies one exact manufacturer part number. A part remains
`candidate` until its exact template version has passed the required hardware
tests. Metadata alone is enough for input checks and mechanical previews; full
manufacturing output additionally requires `template_dir`.

The template directory must contain:

- `template.json` naming the schematic, PCB and adapter executable;
- a reviewed KiCad 9 schematic and PCB;
- an adapter that accepts `design-input.json` and the destination PCB path,
  applies the confirmed outline/terminal geometry and deterministic routing,
  and exits non-zero when it cannot satisfy constraints.

An optional `IC_RESOLVER_ENDPOINT` may return structured JSON candidates for
models not present locally. Search-result HTML is never accepted as electrical
metadata.
