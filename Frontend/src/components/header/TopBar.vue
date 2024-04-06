<template>
  <nav class="navbar navbar-expand-lg fixed-top" :class="[`navbar-${theme}`, `bg-${theme}`]">
    <div class="container-fluid">
      <a class="navbar-brand" href="#">Navbar</a>
      <button
        class="navbar-toggler"
        type="button"
        data-bs-toggle="collapse"
        data-bs-target="#navbarNav"
        aria-controls="navbarNav"
        aria-expanded="false"
        aria-label="Toggle navigation"
        @click="toggleTheme()"
      >
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav">
          <li class="nav-item">
            <a class="nav-link active" aria-current="page" href="#">Home</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#">Features</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#">Pricing</a>
          </li>
          <li class="nav-item">
            <a class="nav-link disabled">Disabled</a>
          </li>
        </ul>
      </div>
      <form>
        <button type="button" :class="['btn', `btn-outline-${btn_theme}`]" @click="toggleTheme()">
          Toggle Theme
          <!-- <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            fill="currentColor"
            class="bi bi-circle"
            viewBox="0 0 16 16"
          >
            <path
              d="M7.988 12.158c-1.851 0-2.941-1.57-2.941-3.99V7.84c0-2.408 1.101-3.996 2.965-3.996 1.857 0 2.935 1.57 2.935 3.996v.328c0 2.408-1.101 3.99-2.959 3.99ZM8 4.951c-1.008 0-1.629 1.09-1.629 2.895v.31c0 1.81.627 2.895 1.629 2.895s1.623-1.09 1.623-2.895v-.31c0-1.8-.621-2.895-1.623-2.895Z"
            />
            <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0ZM1 8a7 7 0 1 0 14 0A7 7 0 0 0 1 8Z" />
          </svg> -->
        </button>
      </form>
    </div>
  </nav>
</template>

<script setup>
import { ref } from 'vue'
let theme = ref('light')
let btn_theme = ref('dark')

function toggleTheme() {
  if (theme.value === 'dark') {
    theme.value = 'light'
    btn_theme.value = 'dark'
  } else {
    theme.value = 'dark'
    btn_theme.value = 'light'
  }
}
</script>

<style scoped></style>
