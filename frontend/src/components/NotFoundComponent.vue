<template>
  <div>
    <img
      :src="`/assets/404.png`"
      alt="404">
  </div>
</template>
<script>
export default {
  name: 'NotFoundComponent'
}
</script>

<style scoped>
img{
  width: 100%;
}
@media (max-width: 1000px){
div{
  background: #DEF2FF;
  width: 100vw;
  height: 80vh;
}
}
</style>
