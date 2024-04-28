<script setup>
import TopBar from "@/components/header/TopBar.vue";
import BodyPart from "@/components/body/BodyPart.vue";
</script>

<template>
  <header>
    <TopBar />
  </header>

  <main>
    <BodyPart />
    <el-backtop :right="30" :bottom="30" />
  </main>
</template>

<style scoped>
* {
  font-family: Helvetica, sans-serif;
}
</style>
