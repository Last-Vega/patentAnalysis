<template>
<v-app>
  <!-- <highcharts :options="options" ref="chart" style="height: 80vh"></highcharts> -->
    <div>
    <div class="text-center top-right">
      <v-container>
        <v-row>
          <v-col>
            <v-checkbox
              v-model="checkbox"
              :label="`ラベル全表示`"
              @click="$emit('change-label', checkbox)"
              style="margin-top: 0px;"
            ></v-checkbox>
          </v-col>

          <v-col>
            <v-btn-toggle value="move">
              <v-btn value="move" small @click="$emit('toggle')">
                Move
              </v-btn>
              <v-btn value="zoom" small @click="$emit('toggle')">
                Zoom
              </v-btn>
            </v-btn-toggle>
          </v-col>
        </v-row>
      </v-container>
    </div>

    <highcharts :options="options" ref="chart" class="center"></highcharts>
  </div>
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
  emits: ['toggle', 'change-label'],
  components: {
  },
  data () {
    return {
      CCContrib: '',
      CTContrib: '',
      ccPath: this.companyItems,
      ctPath: this.termItems,
      checkbox: false
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
.top-right{
  position: absolute;
  top: -1rxm;
  right: 1rem;
  z-index: 5;
}
.center {
  justify-content: center;
  display: flex;
  align-items: center;
}
</style>
