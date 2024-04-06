<template>
  <!-- <div
    class="container"
    style="position: absolute; top: 40%; left: 50%; transform: translate(-50%, -50%)"
  >
    <div class="row g-3">
      <div class="col-md-8">
        <form>
          <div class="mb-3">
            <label for="exampleFormControlTextarea">Example textarea</label>
            <textarea
              class="form-control"
              id="exampleFormControlTextarea"
              rows="3"
              placeholder="Enter text here"
              v-model="testText"
            ></textarea>
          </div>

          <button
            class="btn btn-primary"
            type="submit"
            @click="submitText"
            style="position: relative; left: 100%; transform: translateX(-100%)"
          >
            Submit
          </button>
        </form>
        <div>
          <p>Response: {{ res }}</p>
        </div>
      </div>
    </div>
  </div> -->

  <div>
    <textarea
      v-model="testText"
      placeholder="Enter text here"
      style="width: 100%; height: 200px; margin-bottom: 300px"
    ></textarea>
    <button
      type="button"
      class="btn btn-primary"
      style="position: relative; left: 100%; transform: translateX(-100%); bottom: 400px"
      @click="submitText"
    >
      Submit
    </button>

    <div>
      <p>Response: {{ res.text }}</p>
      <p>Status: {{ res.status }}</p>
      <p>Headers: {{ res.headers }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const testText = ref('')

const res = ref({
  text: null,
  status: null,
  headers: null
})

const submitText = async () => {
  try {
    console.log(testText.value)
    const response = await axios.post('/api/text', { text: testText.value })
    testText.value = ''
    res.value.text = response.data.text
    res.value.status = response.status
    res.value.headers = response.headers

    if (response.status === 200) {
      console.log('Success')
      console.log(response.headers)
      alert(response.data.text)
    } else {
      alert('Error Occurred where status is not 200')
    }
  } catch (error) {
    console.error(error)
    console.error(error.message)
    alert(error)
  }
}
</script>
