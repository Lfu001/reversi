const DiskColor = {
  Dark: 'Dark',
  Light: 'Light',
} as const

type DiskColor = typeof DiskColor[keyof typeof DiskColor]

export { DiskColor }
