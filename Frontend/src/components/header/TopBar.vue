<script setup>
import { ref } from "vue";
const activeIndex = ref("1");
const handleSelect = (index) => {
  activeIndex.value = -1;
  console.log(index);
};
</script>

<template>
  <el-menu
    :default-active="activeIndex"
    class="el-menu-demo"
    mode="horizontal"
    :ellipsis="false"
    @select="handleSelect"
  >
    <el-menu-item
      index="0"
      class="is-active"
      style="width: 40px; margin-left: 15px"
    >
      <img
        style="width: 40px; height: 78px"
        src="@/assets/logo.svg"
        alt="Lab logo"
      />
    </el-menu-item>
    <div class="flex-grow">
      <el-menu-item index="1" style="font-size: x-large"
        >FACE Demo</el-menu-item
      >
    </div>
  </el-menu>
</template>
