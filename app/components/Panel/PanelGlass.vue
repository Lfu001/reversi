<template>
  <div class="container mx-auto flex justify-center">
    <div class="h-screen w-full content-center p-4">
      <div
        class="flex h-full w-full justify-center rounded-3xl bg-emerald-50/30 p-4 drop-shadow-lg backdrop-blur-xl md:max-h-200 dark:bg-emerald-950/30"
      >
        <slot />
      </div>
    </div>
  </div>
</template>
