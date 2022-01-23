<template>
  <v-dialog v-model="flag" fullscreen>
    <v-container fluid fill-height class="loading_bg">
      <v-layout justify-center align-center>
       <v-progress-circular
        :size="70"
        :width="7"
        color="purple"
        indeterminate
        >
        </v-progress-circular>
      </v-layout>
    </v-container>
  </v-dialog>
</template>

<script>
export default {
  name: 'Loading',
  props: {
    flag: {
      type: Boolean,
      required: true
    }
  }
}
</script>

<style lang="scss" scoped>
.loading_bg {
    position: absolute;
    top:0px;
    left:0px;
    width:100%;
    background-color: #000;
    opacity: 0.5;
    z-index: 999;
}
</style>
