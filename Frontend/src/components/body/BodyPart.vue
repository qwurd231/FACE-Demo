<script setup>
import { ref, onUpdated } from "vue";
import axios from "axios";
import Plotly from "plotly.js-dist";
import pako from "pako";

const text = ref("");
const result = ref({
  text: null,
  frequency: null,
  spectra: null,
});
const show = ref(false);
const radio = ref("FACE");

function convertToResult(compressedData, index) {
  let charCodes = [];
  let l = Object.keys(compressedData).length;

  // Iterate over the characters of the text string
  for (let i = 0; i < l; i++) {
    // Get the character code for each character
    charCodes.push(compressedData[i]);
  }

  // Create a Uint8Array from the character codes
  let bytes_text = new Uint8Array(charCodes);

  // Convert the Uint8Array to a string
  let t = pako.ungzip(bytes_text);
  let result_data = new TextDecoder().decode(t);
  if (index !== 0) {
    result_data = result_data
      .substring(1, result_data.length - 1)
      .split(",")
      .map(Number);
  }
  return result_data;
}

async function submit() {
  console.log(text);
  const isWhitespaceString = (str) => !str.replace(/\s/g, "").length;
  if (isWhitespaceString(text.value)) {
    alert("Please enter some text.");
    document.querySelector("textarea").value = "";
    text.value = "";
    return;
  }
  console.log(text.value);
  let gzip_text = pako.gzip(text.value);
  console.log(gzip_text);
  console.log(typeof gzip_text);
  try {
    const response = await axios.post(
      "/api/text",
      {
        text: gzip_text,
      },
      {
        headers: {
          "Content-Encoding": "gzip",
        },
      }
    );
    document.querySelector("textarea").value = "";
    text.value = "";
    console.log(response);
    if (response.status === 200) {
      result.value.text = convertToResult(response.data.text, 0);
      console.log(result.value.text);

      result.value.frequency = convertToResult(response.data.frequency, 1);
      //result.value.frequency = response.data.frequency;
      console.log(result.value.frequency);

      result.value.spectra = convertToResult(response.data.spectra, 1);
      //result.value.spectra = response.data.spectra;
      console.log(result.value.spectra);

      show.value = true;
      console.log(result);
      console.log(response.headers);
    } else {
      console.log(response.status);
      alert(response.statusText);
    }
  } catch (error) {
    console.error(error);
    alert(error);
  }
}

const createGraph = () => {
  const Graph = document.getElementById("graph");
  if (Graph) {
    console.log(result.value.frequency);
    console.log(result.value.spectra);
    Plotly.newPlot(
      Graph,
      [
        {
          x: result.value.frequency,
          y: result.value.spectra,
          mode: "lines",
          type: "scatter",
          name: "estimator",
        },
      ],
      {
        margin: { t: 0 },
        xaxis: { title: "frequency" },
        yaxis: { title: "spectra" },
      }
    );
  } else {
    console.log("Plot container is " + Graph);
  }
};

onUpdated(() => {
  if (show.value) createGraph();
});
</script>

<template>
  <el-radio-group
    v-model="radio"
    size="large"
    style="display: flex; justify-content: center; margin-top: 3%"
  >
    <el-radio-button label="FACE" value="FACE" />
    <el-radio-button label="FACE II" value="FACE II" />
  </el-radio-group>
  <el-input
    v-model="text"
    style="
      width: 70%;
      left: 50%;
      transform: translateX(-50%);
      margin-top: 2%;
      font-size: larger;
    "
    :autosize="{ minRows: 8, maxRows: 11 }"
    resize="none"
    type="textarea"
    maxlength="5000"
    show-word-limit
    placeholder="Please input your text here"
    @keydown.enter.exact="submit"
  />
  <div>
    <el-button
      type="success"
      round
      @click="submit"
      style="
        position: relative;
        transform: translate(-100%, 110%);
        left: 85%;
        margin-top: -2.5%;
      "
      >Submit</el-button
    >
  </div>

  <div v-if="show" style="margin-bottom: max(10%, 80px)">
    <div
      style="
        width: 70%;
        height: 30%;
        margin-left: 14.5%;
        margin-right: 14.5%;
        margin-top: 10%;
        padding: max(0.5%, 10px);
        border: 5px solid aqua;
        border-radius: 25px;
      "
    >
      <h2 style="text-align: center; font-size: x-large">Analysed Text</h2>
      <p
        style="
          font-size: x-large;
          -ms-word-break: break-all;
          word-break: break-all;
          word-break: break-word;
          -webkit-hyphens: auto;
          -moz-hyphens: auto;
          -ms-hyphens: auto;
          hyphens: auto;
        "
      >
        {{ result.text }}
      </p>
    </div>
    <div
      id="graph"
      style="
        width: 70%;
        height: 30%;
        margin-left: 14.5%;
        margin-right: 14.5%;
        margin-top: 3%;
        padding: max(0.5%, 10px);
        border: 5px solid aqua;
        border-radius: 25px;
      "
    ></div>
  </div>
  <div v-else>
    <el-empty description="Analysis results will be displayed here." />
  </div>
</template>
