import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

export const useDataStore = defineStore('main', () => {
  const data = ref({
    text: null,
    frequency: null,
    spectra: null
  })

  async function submitText(testText) {
    try {
      console.log(testText)
      console.log(typeof testText)
      const response = await axios.post('/api/text', { text: testText })
      console.log(response)

      if (response.status === 200) {
        data.value.text = response.data.text
        data.value.frequency = response.data.frequency
        data.value.spectra = response.data.spectra
        console.log(data)
        console.log(response.headers)
      } else {
        console.log(response.status)
        alert(response.statusText)
      }
    } catch (error) {
      console.error(error)
      alert(error)
    }
  }

  return {
    data,
    submitText
  }
})
