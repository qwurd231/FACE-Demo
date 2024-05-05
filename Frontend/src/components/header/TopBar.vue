<script setup>
import { ref } from "vue";
const activeIndex = ref("-1");
const handleSelect = () => {
  activeIndex.value = -1;
};

const value1 = ref(true);
</script>

<template>
  <el-menu
    :default-active="activeIndex"
    class="el-menu-demo"
    mode="horizontal"
    menu-trigger="hover"
    :ellipsis="false"
    @select="handleSelect"
  >
    <el-menu-item
      index="0"
      class="el-menu-item"
      style="
        width: 30px;
        margin-left: 15px;
        margin-top: 5px;
        pointer-events: none;
      "
    >
      <img
        style="width: 50px; height: 50px"
        src="@/assets/logo.png"
        alt="Lab logo"
      />
    </el-menu-item>
    <div>
      <el-menu-item
        class="el-menu-item"
        index="1"
        style="
          font-size: x-large;
          margin-left: 15px;
          height: 49.1px;
          pointer-events: none;
        "
      >
        FACE
      </el-menu-item>
    </div>
    <div style="position: absolute; right: 100px">
      <el-menu-item
        class="el-menu-item is-active"
        id="theme-switcher"
        index="2"
        style="height: 49.1px"
      >
        <el-switch
          v-model="value1"
          style="--el-switch-on-color: gainsboro; --el-switch-off-color: black"
        >
          <template #active-action>
            <span class="custom-active-action">
              <img
                style="
                  width: 14px;
                  height: 14px;
                  margin-bottom: 3px;
                  pointer-events: none;
                "
                src="@/assets/sun.svg"
                role="day mode"
              />
            </span>
          </template>
          <template #inactive-action>
            <span class="custom-inactive-action">
              <img
                style="
                  width: 14px;
                  height: 14px;
                  margin-bottom: 3px;
                  border-radius: 50%;
                  pointer-events: none;
                "
                src="@/assets/moon.svg"
                role="night mode"
              />
            </span>
          </template>
        </el-switch>
      </el-menu-item>
    </div>
    <div style="position: absolute; right: 0px">
      <el-sub-menu class="el-menu-item" index="3" style="padding: 0">
        <template #title>
          <img
            style="
              width: 30px;
              height: 30px;
              margin-top: 10px;
              margin-right: -5px;
            "
            src="@/assets/language.svg"
            alt="language"
          />
        </template>
        <el-menu-item index="3-1">
          <img
            style="width: 15px; height: 15px"
            src="@/assets/link.svg"
            alt="link"
          />
          <p
            style="
              position: relative;
              left: 40%;
              transform: translateX(-40%);
              font-size: large;
            "
          >
            English
          </p>
        </el-menu-item>
        <el-menu-item index="3-2">
          <img
            style="width: 15px; height: 15px"
            src="@/assets/link.svg"
            alt="link"
          />
          <p
            style="
              position: relative;
              left: 40%;
              transform: translateX(-40%);
              font-size: large;
            "
          >
            中文
          </p>
        </el-menu-item>
      </el-sub-menu>
    </div>
  </el-menu>
</template>

<style scoped>
.el-menu-demo {
  height: 50px;
  position: absolute;
  width: 100%;
  top: 0;
  padding-top: 0;
  padding-bottom: 0;
}
.el-menu-item {
  height: 40px;
}
.el-menu-item:hover {
  background-color: #f0f0f0;
  color: black;
}
</style>
