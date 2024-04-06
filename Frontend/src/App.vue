<script setup>
import TopBar from './components/header/TopBar.vue'
import TestTextArea from './components/body/TestTextArea.vue'
import StatisticGraph from './components/body/StatisticGraph.vue'
import StatisticExplanation from './components/body/StatisticExplanation.vue'
</script>

<template>
  <TopBar />
  <div class="main_frame">
    <TestTextArea />
    <div class="module" style="position: absolute; top: 60%">
      <StatisticGraph />
      <StatisticExplanation />
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
