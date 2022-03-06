<template>
<v-app>
  <highcharts :options="options" ref="chart"></highcharts>
  <v-container class="grey lighten-5">
    <v-row no-gutters>
        <v-col cols="12" sm="12">
          企業の配置に強く影響を与えているのは<span class="relationC">{{companyItems}}</span>の関係です
        </v-col>
        <v-col cols="12" sm="2"> </v-col>
        <v-col cols="12" sm="12">
          用語の配置に強く影響を与えているのは<span class="relationT">{{termItems}}</span>の関係です
        </v-col>
    </v-row>
    </v-container>
  </v-app>
</template>

<script>

export default {
  name: 'DisplayLatentSpace',
  props: {
    options: {
      type: Object,
      required: true
    },
    companyItems: {
      type: String,
      required: true
    },
    termItems: {
      type: String,
      required: true
    }
  },
  components: {
  },
  data () {
    return {
      CCContrib: '',
      CTContrib: '',
      ccPath: this.companyItems,
      ctPath: this.termItems
    }
  },
  methods: {

  }

}
</script>

<style scoped>
.relationC {
  color: blue;
}
.relationT {
  color: red;
}
</style>
