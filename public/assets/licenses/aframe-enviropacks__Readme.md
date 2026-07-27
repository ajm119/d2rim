# AFRAME Enviropacks

A simple way of adding high-definition-looking environments and materials to your A-Frame project.

- HDRIs from [HDRI Haven](https://hdrihaven.com/) licensed CC0
- Textures from [Texture Haven](https://texturehaven.com/) and [CC0 Textures](https://cc0textures.com/) licensed CC0. Compressed with [Squoosh](https://squoosh.app/editor)
- Models either from [CC0 Textures](https://cc0textures.com/) or created by [Zach Capalbo](https://zachcapalbo.com)
- Source code licensed MIT license
- Matcaps created by [Zach Capalbo](https://zachcapalbo.com), licensed CC0

Include the js file in the header:

```
<script src="https://cdn.jsdelivr.net/npm/aframe-enviropacks@latest/aframe-enviropacks.js"></script>
```

then add the `enviropack` component to an entity and you're ready to go!

```
<a-entity enviropack="preset: tankfarm"></a-entity>
```

![Enviropacks Preview Screenshots](https://gitlab.com/zach-geek/aframe-enviropacks/-/raw/dev/enviropacks.gif?inline=false)

You can view the full example on Glitch:

[https://glitch.com/edit/#!/aframe-enviropacks?path=index.html](https://glitch.com/edit/#!/aframe-enviropacks?path=index.html)

Or check out this "Pumpkin Chunkin" demo to see it in a "real world" scenario:

https://gilded-rumbling-rib.glitch.me/

# Components

---

## `enviropack`

Sets up a whole preset environment all in one component.

| Property | Description |
|------|------|
| preset | Preset environment to choose from. One of:  `tankfarm`, `sandstone`, `interior`, `winter`, `autumn`, or `night`|
| baseUrl | Base URL for assets. If left blank will be guessed based on the included `.js` file location |

---

## `enviropack-material`

Applies one of the materials included in the pack.

| Property | Description |
|------|------|
| material | Name of preset material to use. One of: `ground-grass-rock`, `ground-sandstone`, `ground-wood-floor`, `ground-snow`, `ground-forest`, `bark`, `gold`, `metal`, `plastic`, `fabric`, `concrete`, `rock`, `wood`, `planks`, `snow`. Or could also be a "raw" material: `Fabric026`, `Metal009`, `Metal035`, `Plastic`, `Snow003`, `Snow004`, `Wood027`, `WoodFloor041`, `aerial_grass_rock`, `bark_brown_02`, `brown_planks_04`, `forrest_ground_03`, `mossy_rock`, `rock_05`, `sandstone_cracks` |
| autoApply | If true, it will replace any material in the entity's mesh. This is useful for replacing the material on a loaded GLTF file, for instance.|
| shader | Force a specific shader. Defaults to `auto`, which will chose `standard` on desktop, and `pbmatcap` on mobile |

---

## `enviropack-hdri`

| Property | Description |
|------|------|
| hdri | Name of preset HDRI. One of: `abandoned_tank_farm_02`, `autumn_hockey`, `large_corridor`, `the_sky_is_on_fire`, `winter_lake_01`, `dikhololo_night_edit` |
| backstop | If specified, will use this image as the sky background image (but not for lighting or envMap) |

---

# Physically Based Matcaps

This package registers the `pbmatcap` shader, which creates a "Physically Based"
matcap material. The `pbmatcap` shader uses material textures (such as roughnessMap and metalnessMap textures) to mix between custom matcap textures created for each different environment.

`pbmatcap` shader can approximate the full THREE.js `standard` shader while giving performance acceptable to a mobile device. By default, the enviropack components will automatically switch to `pbmatcap` on mobile, or `standard` on
desktop. This behavior can be overriden with the `shader` property of the `enviropack-material` system.
