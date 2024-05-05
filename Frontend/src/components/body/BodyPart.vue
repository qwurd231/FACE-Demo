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
});
const show = ref(false);
const radio = ref("FACE");

function clean() {
  document.querySelector("textarea").value = "";
  text.value = "";
  result.value.text = [];
  result.value.color = [];
  result.value.frequency = [];
  result.value.spectra = [];
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

  var result = result_data.split("</>");

  return result;
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
    /^[A-Za-z0-9,-–—−—―›~″•±×.:;/^#@™®%&`…|€$!?_=+\-$*''"(){}<>\s[\]\\]*$/.test(
      str
    );

  // test if string contains "<*/>", where * is any character, use regex
  //const isHTML = (str) => /<.*\/>/.test(str);

  if (!isEnglish(str)) {
    alert("Please enter English text only.");
    valid = false;
    clean();
    return valid;
  } else {
    return valid;
  }
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
        newText = newText.trimRight() + " ";
      }
      newlineCount = 0;
    }
    newText += str[i];
  }
  return newText;
}

async function submit() {
  console.log(text);

  let valid = true;

  if (radio.value === "FACE") {
    valid =
      checkEnglishText(text.value, valid) &&
      checkWhitespaceString(text.value, valid);
  }

  if (!valid) {
    show.value = false;
    return;
  }
  let sentText = removeNewLine(text.value);
  console.log(sentText);
  console.log(sentText.length);
  let gzip_text = pako.gzip(sentText);
  console.log(gzip_text);
  console.log(gzip_text.length);

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

      clean();
      console.log(response);

      if (response.status === 200) {
        result.value.text = gzipToResult(response.data)[0]
          .substring(5)
          .split("の");
        result.value.color = gzipToResult(response.data)[1]
          .substring(6)
          .split(",");
        result.value.frequency = gzipToResult(response.data)[2]
          .substring(10)
          .split(",");
        result.value.spectra = gzipToResult(response.data)[3]
          .substring(8)
          .split(",");

        console.log(result.value.text);

        console.log(result.value.color);

        console.log(result.value.frequency);

        console.log(result.value.spectra);

        show.value = true;
        console.log(result);
        console.log(response.headers);
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
        result.value.text = gzipToResult(response.data)[0]
          .substring(5)
          .split("の");
        result.value.color = gzipToResult(response.data)[1]
          .substring(6)
          .split(",");
        result.value.frequency = gzipToResult(response.data)[2]
          .substring(10)
          .split(",");
        result.value.spectra = gzipToResult(response.data)[3]
          .substring(8)
          .split(",");

        console.log(result.value.text);

        console.log(result.value.color);

        console.log(result.value.frequency);

        console.log(result.value.spectra);

        show.value = true;
        console.log(result);
        console.log(response.headers);
      } else {
        console.log(response.status);
        alert(response.statusText);
      }
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
          line: { shape: "spline", smoothing: 1.3 },
          type: "scatter",
          name: "first plot",
          showlegend: true,
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
    maxlength="10000"
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
      <h2 style="text-align: center; font-size: xx-large">Analysed Text</h2>
      <div
        v-for="(item, index) in result.text"
        :key="index"
        style="padding: 0.5% 3%"
      >
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
          "
          :style="{ backgroundColor: result.color[index] }"
        >
          {{ item }}
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
        border: 5px solid aqua;
        border-radius: 25px;
      "
    >
      <div id="graph"></div>
    </div>
  </div>
  <div v-else>
    <el-empty description="Analysis results will be displayed here." />
  </div>
</template>
