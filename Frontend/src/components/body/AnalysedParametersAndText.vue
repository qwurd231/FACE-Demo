<template>
  <div
    class="container"
    style="display: flex; flex-direction: column; align-items: center; margin-bottom: 10%"
  >
    <h1>Statistic Graph</h1>
    <div class="card">
      <h5
        v-if="data.text"
        class="card-title"
        style="display: flex; justify-content: center; margin-top: 2%; margin-bottom: 2%"
      >
        Spectra of Your Text
      </h5>
      <div v-else>
        <h4 style="display: flex; justify-content: center; margin-top: 2%; margin-bottom: 2%">
          Please enter some text to see the analysis
        </h4>
      </div>
      <div id="graph" style="width: 80%; height: 80%; display: flex; justify-content: center"></div>
    </div>
    <div v-if="data.text" class="card" style="margin-top: 5%">
      <h5 class="card-text" style="width: 96%; margin: 2%; text-align: left">
        {{ data.text }}
      </h5>
    </div>
  </div>
</template>

<script setup>
import { onUpdated, ref } from 'vue'
import Plotly from 'plotly.js-dist'
import { useDataStore } from '../../store/index'

const store = useDataStore()
const data = ref(store.data)

const createGraph = () => {
  const Graph = document.getElementById('graph')
  if (Graph) {
    Plotly.newPlot(
      Graph,
      [
        {
          x: data.value.frequency,
          y: data.value.spectra,
          module: 'lines',
          type: 'scatter',
          name: 'estimator',
          line: { color: 'burlywood' }
        }
      ],
      {
        showlegend: true,
        margin: { t: 0 },
        xaxis: { title: 'frequency' },
        yaxis: { title: 'spectra' }
      }
    )
  } else {
    console.log('Plot container is ' + Graph)
  }
}

// Call createGraph function when component is mounted
onUpdated(() => {
  createGraph()
})
</script>

<style scoped>
.card {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 90%;
}
</style>
