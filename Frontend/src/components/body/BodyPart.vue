<script setup>
import { ref, onUpdated } from "vue";
import axios from "axios";
import Plotly from "plotly.js-dist";
import pako from "pako";

const text = ref("");
const result = ref({
  text: [],
  color: [],
  frequency: [],
  spectra: [],
  model_text: '',
  model_frequency: [],
  model_spectra: [],
  model_fittedvalues: [],
  prompt_length: 0,
  text_fittedvalues: [],
});
const show = ref(0);
const radio = ref("FACE");

function clean() {
  document.querySelector("textarea").value = "";
  text.value = "";
  show.value = 1;
  result.value.text = [];
  result.value.color = [];
  result.value.frequency = [];
  result.value.spectra = [];
  result.value.model_text = '';
  result.value.model_frequency = [];
  result.value.model_spectra = [];
  return;
}

function gzipToResult(compressedData) {
  console.log(compressedData);
  let charCodes = [];
  let l = Object.keys(compressedData).length;

  // Iterate over the characters of the text string
  for (let i = 0; i < l; i++) {
    // Get the character code for each character
    charCodes.push(compressedData[i]);
  }
  console.log(charCodes);

  // Create a Uint8Array from the character codes
  let bytes_text = new Uint8Array(charCodes);

  //
  let t = pako.ungzip(bytes_text);
  let result_data = new TextDecoder().decode(t);

  var res = result_data.split("け");

  return res;
}

function checkWhitespaceString(str, valid) {
  const isWhitespaceString = (str) => !str.replace(/\s/g, "").length;
  const isNewLine = (str) => !str.replace(/\n/g, "").length;
  if (isWhitespaceString(str) || isNewLine(str)) {
    alert("Please enter some text.");
    valid = false;
    clean();
    return valid;
  } else {
    return valid;
  }
}

function checkEnglishText(str, valid) {
  const isEnglish = (str) =>
    /^[A-Za-z0-9,-–—−—―›~″•±×\.:;\/^#@™®%&`…\|€\$!\?_=\+\-$*''"‘’“”()\{\}<>\s\[\]\\]*$/.test(
      str
    );

  // test if string contains "<*/>", where * is any character, use regex
  //const isHTML = (str) => /<.*\/>/.test(str);

  if (!isEnglish(str)) {
    alert("Please enter English text only.");
    valid = false;
    clean();
  }
  return valid;
}

function removeNewLine(str) {
  let newText = "";
  let newlineCount = 0;

  for (let i = 0; i < str.length; i++) {
    if (str[i] === "\n") {
      newlineCount++;
      if (newlineCount === 3) {
        console.log("Removed extra new line");
        continue; // Skip adding this newline character
      }
    } else {
      if (newlineCount === 1) {
        newText = newText.trimEnd() + " ";
      }
      newlineCount = 0;
    }
    newText += str[i];
  }
  return newText;
}

function result_split(response_data) {
  let res = gzipToResult(response_data);
  console.log(res);
  result.value.text = res[0]
          .substring(5)
          .split("分");
  result.value.color = res[1]
    .substring(6)
    .split(",");
  result.value.frequency = res[2]
    .substring(10)
    .split("る")
    .map(function (x) {
      return x.split(",").map(Number);
    })
  result.value.spectra = res[3]
    .substring(8)
    .split("る")
    .map(function (x) {
      return x.split(",").map(Number);
    })
  result.value.model_text = res[4]
    .substring(69);
  result.value.model_frequency = res[5]
    .substring(16)
    .split(",").map(Number);
  result.value.model_spectra = res[6]
    .substring(14)
    .split(",").map(Number);
  result.value.model_fittedvalues = res[7]
    .substring(19)
    .split(",").map(Number);
  result.value.prompt_length = Number(res[8].substring(14));
  /*result.value.text_fittedvalues = res[9]
    .substring(19)
    .split(",").map(Number);*/

  console.log('result_text:', result.value.text);

  console.log('result_color:', result.value.color);

  console.log('result_frequency:', result.value.frequency);

  console.log('result_spectra:', result.value.spectra);

  console.log('result_model_text:', result.value.model_text);

  console.log('result_model_frequency:', result.value.model_frequency);

  console.log('result_model_spectra:', result.value.model_spectra);

  console.log('result_model_fittedvalues:', result.value.model_fittedvalues);

  console.log('result_prompt_length:', result.value.prompt_length);

  //console.log('result_text_fittedvalues:', result.value.text_fittedvalues);

  console.log('result:', result.value);
}

async function submit() {
  if (show.value === 1) {
    alert("Please wait for the current analysis to complete.");
    return;
  }

  console.log(text);

  let valid = true;

  if (radio.value === "FACE") {
    valid =
      checkEnglishText(text.value, valid) &&
      checkWhitespaceString(text.value, valid);
  }

  if (!valid) {
    show.value = 0;
    return;
  }
  let sentText = removeNewLine(text.value);
  console.log(sentText);
  console.log(sentText.length);
  let gzip_text = pako.gzip(sentText);
  console.log(gzip_text);
  console.log(gzip_text.length);
  clean();

  try {
    let response;
    if (sentText.length > gzip_text.length) {
      response = await axios.post(
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

      console.log(response);

      if (response.status === 200) {
        result_split(response.data);

        show.value = 2;
      } else {
        console.log(response.status);
        alert(response.statusText);
      }
    } else {
      response = await axios.post("/api/text", {
        text: sentText,
      });

      clean();
      console.log(response);

      if (response.status === 200) {
        result_split(response.data);

        show.value = 2;
      } else {
        console.log(response.status);
        alert(response.statusText);
      }
    }
  } catch (error) {
    console.error(error);
    alert(error);
    if (error.response.status === 429) {
      show.value = 0;
    }
  }
}

const createGraph = () => {
  const Graph = document.getElementById("origin-graph");
  if (Graph) {
    console.log(result.value.frequency);
    console.log(result.value.spectra);
    var data = [];
    for (let i = 0; i < result.value.frequency.length; i++) {
      data.push({
        x: result.value.frequency[i],
        y: result.value.spectra[i],
        type: "scatter",
        mode: "lines",
        line: { shape: "spline", smoothing: 1.3 },
        name: "Spectra " + (i + 1),
      });
    }
    Plotly.newPlot(Graph, data, {
      margin: { t: 0 },
      xaxis: { title: "frequency" },
      yaxis: { title: "spectra" },
    });
  } else {
    console.log("Plot container is " + Graph);
  }

  const ProcessedGraph = document.getElementById("processed-graph");
  if (ProcessedGraph) {
    console.log(result.value.model_frequency);
    console.log(result.value.model_spectra);
    console.log(result.value.model_fittedvalues);
    var data = [];
    data.push({
      x: result.value.model_frequency,
      y: result.value.model_spectra,
      type: "scatter",
      mode: "lines",
      line: { shape: "spline", smoothing: 1.3 },
      name: "Model",
    });
    /*let text_frequency = result.value.frequency.flat().sort((a, b) => a - b);
    console.log('text_frequency:', text_frequency);
    data.push({
      x: text_frequency,
      y: result.value.text_fittedvalues,
      type: "scatter",
      mode: "lines",
      line: { shape: "spline", smoothing: 1.3 },
      name: "Text Fitted Values",
    });*/
    data.push({
      x: result.value.model_frequency,
      y: result.value.model_fittedvalues,
      type: "scatter",
      mode: "lines",
      line: { shape: "spline", smoothing: 1.3 },
      name: "Fitted Values",
    });
    Plotly.newPlot(ProcessedGraph, data, {
      margin: { t: 0 },
      xaxis: { title: "model frequency" },
      yaxis: { title: "model spectra" },
    });
  } else {
    console.log("Plot container is " + ProcessedGraph);
  }
};

onUpdated(() => {
  if (show.value === 2) {
    createGraph();
  } else if (show.value === 1) {
    document.querySelector("textarea").value = "";
    text.value = "";
  }
});
// -webkit-mask: linear-gradient(90deg,#000 70%,#0000 0) left/20% 100%;
</script>

<template>
  <el-radio-group
    v-model="radio"
    size="large"
    fill="darkcyan"
    style="display: flex; justify-content: center; margin-top: 120px"
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
    maxlength="1000000"
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
    >
      Submit
    </el-button>
  </div>

  <div v-if="show === 2" style="margin-bottom: max(10%, 80px)">
    <div
      style="
        width: 70%;
        height: 30%;
        margin-left: 14.5%;
        margin-right: 14.5%;
        margin-top: max(8%, 50px);
        padding: max(0.5%, 10px);
        border: 5px solid gainsboro;
        border-radius: 25px;
      "
    >
      <h2 style="text-align: center; font-size: xx-large">Analysed Text</h2>
      <div
        v-for="(item, index) in result.text"
        :key="index"
        style="padding: 0% 3%"
      >
        <p
          style="
            font-size: x-large;
            font-size: x-large;
            -ms-word-break: break-all;
            word-break: break-all;
            word-break: break-word;
            -webkit-hyphens: auto;
            -moz-hyphens: auto;
            -ms-hyphens: auto;
            hyphens: auto;
            margin-left: 3%;
          "
          :style="{ backgroundColor: result.color[index] }"
        >
          <span 
            style="text-align: end;"
            :style="{ backgroundColor: result.color[index] }"
          >
            {{ index + 1 }}
          </span>

          {{ item }}
      </p>
      </div>
    </div>
    <div
      style="
        width: 70%;
        height: 30%;
        margin-left: 14.5%;
        margin-right: 14.5%;
        margin-top: 3%;
        padding: max(0.5%, 10px);
        padding-top: 1%;
        border: 5px solid gainsboro;
        border-radius: 25px;
      "
    >
      <div id="origin-graph"></div>
    </div>
    <div
      style="
        width: 70%;
        height: 30%;
        margin-left: 14.5%;
        margin-right: 14.5%;
        margin-top: max(8%, 50px);
        padding: max(0.5%, 10px);
        border: 5px solid gainsboro;
        border-radius: 25px;
      "
    >
      <h2 style="text-align: center; font-size: xx-large">Model Text</h2>
      <div style="padding: 0% 3%">
        <span
          style="
            font-size: x-large;
            font-size: x-large;
            -ms-word-break: break-all;
            word-break: break-all;
            word-break: break-word;
            -webkit-hyphens: auto;
            -moz-hyphens: auto;
            -ms-hyphens: auto;
            hyphens: auto;
            margin-left: 3%;
            background-color: paleturquoise;
          "
        >
          {{ result.model_text.substring(0, result.prompt_length) }}
        </span>
        <span 
          style="
            background-color: white;
            color: black;
            font-size: x-large;
            font-size: x-large;
            -ms-word-break: break-all;
            word-break: break-all;
            word-break: break-word;
            -webkit-hyphens: auto;
            -moz-hyphens: auto;
            -ms-hyphens: auto;
            hyphens: auto;
            margin-left: 3%;
          ">
            {{ result.model_text.substring(result.prompt_length) }}
        </span>
      </div>
    </div>
    <div
      style="
        width: 70%;
        height: 30%;
        margin-left: 14.5%;
        margin-right: 14.5%;
        margin-top: 3%;
        padding: max(0.5%, 10px);
        padding-top: 1%;
        border: 5px solid gainsboro;
        border-radius: 25px;
      "
    >
      <div id="processed-graph"></div>
    </div>
  </div>
  <div v-else-if="show === 1">
    <div
      style="
        position: relative;
        margin-top: max(8%, 50px);
        margin-bottom: max(5%, 50px);
      "
    >
      <div class="loader"></div>
    </div>
  </div>
  <div v-else>
    <el-empty
      class="empty"
      description="Analysis results will be displayed here." 
    />
  </div>
</template>
<style scoped>
.loader {
  width: 120px;
  height: 20px;
  margin: 0 auto;
  mask: linear-gradient(90deg,#000 70%,#0000 0) left/20% 100%;
  background:
   linear-gradient(rgb(53, 53, 53) 0 0) left -25% top 0 /20% 100% no-repeat
   #ddd;
  animation: l7 1s infinite steps(6);
}
@keyframes l7 {
    100% {background-position: right -25% top 0}
}
</style>
