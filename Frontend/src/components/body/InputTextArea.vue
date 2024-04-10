<template>
  <div
    class="container"
    style="
      display: flex;
      width: 60%;
      height: 30%;
      position: absolute;
      top: 40%;
      left: 50%;
      transform: translate(-50%, -50%);
    "
  >
    <textarea
      class="form-control"
      v-model="testText"
      placeholder="Enter text here"
      style="height: 100%; width: 100%; position: relative; left: 50%; transform: translateX(-50%)"
    ></textarea>
    <button
      type="button"
      class="btn btn-primary"
      style="position: relative; top: 102%; height: 20%; right: 10%; transform: translateX(60%)"
      @click="submit(testText)"
    >
      Submit
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useDataStore } from '../../store/index'

const testText = ref('')

const store = useDataStore()

function submit(text) {
  store.submitText(text)
  testText.value = ''
}
</script>
