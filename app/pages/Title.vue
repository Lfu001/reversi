<template>
  <div>
    <div
      id="water-drop-animation"
      class="water-drop-animation absolute h-screen w-screen overflow-clip bg-neutral-300 transition-colors duration-300 dark:bg-sky-950"
    />
    <div class="container mx-auto flex justify-center">
      <div class="h-screen w-full content-center p-4">
        <div
          class="flex h-full w-full justify-center rounded-3xl bg-emerald-50/30 p-4 drop-shadow-lg backdrop-blur-xl md:max-h-200 dark:bg-emerald-950/30"
        >
          <div class="flex flex-col items-center justify-center">
            <h1
              class="select-none mb-30 content-center text-5xl font-black text-black dark:text-white"
            >
              AIリバーシ
            </h1>
            <div class="flex flex-col items-center justify-center">
              <TextField
                v-model="guestName"
                class="mb-4"
                placeholder="ゲスト名"
              />
              <Button label="次へ →" />
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * The name of the guest who is currently using the app.
 */
const guestName = ref('')

/**
 * The ID of the interval that creates water drops periodically.
 */
const intervalId = ref<number | null>(null)

onMounted(() => {
  intervalId.value = setInterval(createWaterDrop, 400)
})

onUnmounted(() => {
  if (intervalId.value != null) {
    clearInterval(intervalId.value)
  }
})

/**
 * An array of points that are used to create water drops.
 */
const points = [
  [0.007194900816254068, 0.6107369244755445],
  [0.016771439317413896, 0.7386803255291363],
  [0.10811998322664987, 0.8524695625485383],
  [0.15433351978009008, 0.7603044428484877],
  [0.2974485843811922, 0.6587778306149521],
  [0.26348683408189527, 0.4702040759807322],
  [0.15834638881858562, 0.6246782039386432],
  [0.1143658140401739, 0.48139635215811777],
  [0.3509230535415645, 0.7925729536001118],
  [0.14408175308373158, 0.9582830392629815],
  [0.4327137008043226, 0.6055032261385835],
  [0.5675000024734044, 0.6858750550216604],
  [0.3233371811898142, 0.9292482474355859],
  [0.4844661533893571, 0.98873570806225],
  [0.2437138981434943, 0.9959202253784701],
  [0.02926663735676377, 0.9732256053936215],
  [0.012849501268989691, 0.4411826423261444],
  [0.564822841341049, 0.8714982010195961],
  [0.21374355241424448, 0.8429155143432647],
  [0.04315617815609093, 0.3349761870185554],
  [0.4121146377001771, 0.3840203706192425],
  [0.18126675231182873, 0.4022865933447536],
  [0.5200135321293826, 0.7770838643122056],
  [0.4585520190191587, 0.8881614589440223],
  [0.500765894779682, 0.4642453378882798],
  [0.7161068706663951, 0.8722056899848696],
  [0.6306670891644783, 0.5539230472971064],
  [0.005158569592166956, 0.21419336103723285],
  [0.3482321959122049, 0.5256143548235308],
  [0.2883861433179602, 0.2968585535071052],
  [0.6429748639531936, 0.9756951655313528],
  [0.0629483981686556, 0.10952543755096283],
  [0.6166423606228084, 0.43172267086183624],
  [0.11124340933125453, 0.2543137999713118],
  [0.7446614054722126, 0.6176378560488662],
  [0.42023514945800167, 0.7086044228840418],
  [0.19914391634052517, 0.038095986405420945],
  [0.6727241290488807, 0.7132178770507671],
  [0.2711688223682127, 0.14420535753666763],
  [0.41924552832258033, 0.27371660233815376],
  [0.8774037664268681, 0.9023398494723986],
  [0.007009093584780052, 0.8499162445554904],
  [0.8287400774849394, 0.7282750251200728],
  [0.42242591959537823, 0.16060341339759437],
  [0.9693528026571694, 0.8587050469291496],
  [0.9421471796477356, 0.9875721837738938],
  [0.7125640993673237, 0.47704100649486514],
  [0.8698797916490701, 0.4314935796293942],
  [0.749657277676105, 0.36950651682694147],
  [0.5851925627448005, 0.12327190900325809],
  [0.9952065504209086, 0.6990996476171593],
  [0.5067017937073943, 0.3398612556236652],
  [0.6390830153314762, 0.3334981225754955],
  [0.3466825120456735, 0.007268132051578696],
  [0.9306754091331324, 0.5691183338498679],
  [0.7485392046164323, 0.9882651308352797],
  [0.4540805809815499, 0.019668653545447078],
  [0.03399426503983527, 5.5803142328864075e-5],
  [0.8191968506355192, 0.5462326580093574],
  [0.6210228536460071, 0.02714329446175465],
  [0.7329936208977197, 0.2595931275924586],
  [0.16064415566869392, 0.13631596511883842],
  [0.70436930325731, 0.13925430612116702],
  [0.786334537278802, 0.007940373338431065],
  [0.9751931200286669, 0.31723190365211584],
  [0.5218265728228013, 0.24096818370211132],
  [0.9450564423192281, 0.20093460761480195],
  [0.9862488001963614, 0.42906649325039786],
  [0.8488369644744799, 0.31455799518278393],
  [0.9084519747341943, 0.08321055541088938],
  [0.9801372169082583, 0.010888520109560124],
  [0.8038896624023212, 0.11173551006693117],
  [0.6211928945327384, 0.2270960610818986],
]

/**
 * Create a water drop animation.
 */
function createWaterDrop() {
  const waterDropAnimation = document.getElementById('water-drop-animation')
  if (waterDropAnimation == null) {
    return
  }

  const width = waterDropAnimation.clientWidth
  const height = waterDropAnimation.clientHeight

  // Choose a random point from the array of points.
  const index = Math.floor(Math.random() * points.length)
  const point = points[index]
  // Create the water drop element.
  const waterDrop = document.createElement('div')
  waterDrop.classList.add('water-drop')
  // Set random position.
  waterDrop.style.top = `${point[0] * height}px`
  waterDrop.style.left = `${point[1] * width}px`
  // Set random color.
  const hue = Math.floor(Math.random() * 360)
  const saturation = 60 + Math.floor(Math.random() * 30)
  const lightness = 50 + Math.floor(Math.random() * 30)
  waterDrop.style.background = `hsla(${hue}, ${saturation}%, ${lightness}%, 0.8)`
  // Set random size
  const size = 20 + Math.floor(Math.random() * 50)
  waterDrop.style.width = `${size}px`
  waterDrop.style.height = `${size}px`
  // Set random delay time
  const delay = Math.random() * 5
  waterDrop.style.animationDelay = `${delay}s`

  waterDropAnimation.appendChild(waterDrop)

  // Remove water drop after it moves off screen.
  setTimeout(
    () => {
      waterDrop.remove()
    },
    (20 + delay) * 1000,
  )
}
</script>

<style scoped>
.water-drop-animation {
  animation: moveWaterDrops 10s linear infinite;
}

@keyframes moveWaterDrops {
  0% {
    background-position: 0 0;
  }
  100% {
    background-position: 100% 100%;
  }
}

.water-drop-animation :deep(.water-drop) {
  position: absolute;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  box-shadow: 0 0 15px rgba(255, 255, 255, 0.5);
  opacity: 0;
  animation:
    moveDrop 20s linear forwards,
    fadeIn 1s forwards;
}

@keyframes moveDrop {
  0% {
    transform: translateX(0);
  }
  100% {
    transform: translate(-100vw, 100vh);
  }
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
}
</style>
