<template>
  <div class="relative h-screen w-full">
    <BackgroundWaterDrop />

    <div class="relative z-0 h-full w-full">
      <PanelGlass>
        <slot />
      </PanelGlass>
    </div>
  </div>
</template>
