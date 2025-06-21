<template>
  <div>
    <h2
      v-if="title"
      class="text-lg font-medium text-gray-700 dark:text-gray-300 mb-3 transition-colors duration-300"
    >
      {{ title }}
      <span
        v-if="showCount && maxItems > 0"
        class="text-sm ml-2 text-gray-400"
      >
        ({{ items.length }} / {{ maxItems }})
      </span>
    </h2>

    <div v-if="items.length > 0">
      <div
        v-for="(item, index) in items"
        :key="(typeof item === 'string' ? item : item.id) || index"
        class="flex items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md transition-colors duration-300 mb-1 last:mb-0"
      >
        <span class="text-gray-800 dark:text-gray-200 transition-colors duration-300">
          {{ typeof item === 'string' ? item : item.name }}
        </span>
      </div>
    </div>

    <div
      v-else-if="emptyMessage"
      class="text-center py-4 text-gray-500 dark:text-gray-400 transition-colors duration-300"
    >
      {{ emptyMessage }}
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * An item in the list.
 *
 * The item can be an object with any set of properties, but it must have at
 * least one of the following properties:
 *
 * - `name`: a string that will be displayed in the list
 * - `id`: a string or number that will be used as the item's ID
 *
 * If the item is a string, it will be used as the item's name and ID.
 */
interface ListItem {
  /**
   * The item's name, which will be displayed in the list.
   */
  name?: string
  /**
   * The item's ID, which will be used to identify the item.
   */
  id?: string | number
}

interface Props {
  /**
   * Array of items to display in the list
   */
  items: Array<string | ListItem>
  /**
   * Optional title for the list
   */
  title?: string
  /**
   * Maximum number of items (for count display)
   */
  maxItems?: number
  /**
   * Whether to show item count in the title
   */
  showCount?: boolean
  /**
   * Message to display when the list is empty
   */
  emptyMessage?: string
}

withDefaults(defineProps<Props>(), {
  title: '',
  maxItems: 0,
  showCount: true,
  emptyMessage: 'No items',
})
</script>
