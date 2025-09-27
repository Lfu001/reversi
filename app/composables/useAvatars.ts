/**
 * An array of avatar file names.
 */
const fileNames = [
  'browsing-giraffe.png',
  'grazing-sheep.png',
  'majestic-stag.png',
  'soaring-eagle.png',
  'perched-squirrel.png',
  'wise-owl.png',
  'camouflaged-octopus.png',
  'cutting-marlin.png',
  'giant-squid.png',
  'humpback-whale.png',
  'penguin-on-ice.png',
  'leaping-dolphin.png',
  'seal-on-rock.png',
  'ancient-oak.png',
  'autumn-leaves.png',
  'bamboo-grove.png',
  'delicate-fern.png',
  'vibrant-succulent.png',
  'crystalline-dragon.png',
  'forest-spirit-deer.png',
  'mythical-griffin.png',
  'sea-serpent.png',
  'shimmering-phoenix.png',
]

export const useAvatars = () => {
  /**
   * An array of file URLs to the avatar images.
   */
  const fileUrls = fileNames.map(fileName => `/images/avatars/${fileName}`)

  /**
   * Chooses a random avatar file URL from the array of avatar file URLs.
   */
  const choose = () => {
    const randomIndex = Math.floor(Math.random() * fileUrls.length)
    return fileUrls[randomIndex]
  }

  return {
    fileUrls,
    choose,
  }
}
