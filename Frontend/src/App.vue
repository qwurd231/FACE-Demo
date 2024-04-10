<script setup>
import TopBar from './components/header/TopBar.vue'
import InputTextArea from './components/body/InputTextArea.vue'
import AnalysedParametersAndText from './components/body/AnalysedParametersAndText.vue'
</script>

<template>
  <TopBar />
  <div class="main_frame">
    <InputTextArea />
    <div class="module" style="position: absolute; top: 60%; width: 60%">
      <AnalysedParametersAndText />
    </div>
  </div>

  <!-- <RouterView /> -->
</template>

<style scoped>
.body {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  width: 100px;
}

.main_frame {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  margin-bottom: 200px;
}
</style>
