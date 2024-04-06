<template>
  <div class="container">
    <div class="row">
      <h1>Statistic Graph</h1>
      <div class="row">
        <div class="card">
          <div class="card-body">
            <h5 class="card-title">Statistic Graph</h5>
            <!-- Corrected binding of ref -->
            <div id="graph" style="width: 400px; height: 300px"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import Plotly from 'plotly.js-dist'

// Create a ref for the Plotly container

// Function to create the Plotly graph
const createGraph = () => {
  const Graph = document.getElementById('graph')
  if (Graph) {
    Plotly.newPlot(
      Graph,
      [
        {
          x: [1, 2, 3, 4, 5],
          y: [1, 2, 4, 8, 16]
        }
      ],
      {
        margin: { t: 0 }
      }
    )
  } else {
    console.log('Plot container is ' + Graph)
  }
}

// Call createGraph function when component is mounted
onMounted(() => {
  createGraph()
})
</script>

<style scoped>
.card-body {
  display: flex;
  flex-direction: column;
  align-items: center;
}
</style>
