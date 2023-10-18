<template>
  <v-app>
  <Loading :flag="isShow" />
  <comapnyTabel />
  <!-- <v-container>
    熊谷組とコラボする企業を選択してください
      <v-dialog
        v-model="dialog"
        width="500"
      >
      <template v-slot:activator="{ on, attrs }">
      <v-btn
        dark
        v-bind="attrs"
        v-on="on"
        @click="showPrediction"
      >
        推薦結果を見る
      </v-btn>
      </template>
    </v-dialog>
    <v-data-table
            :headers="headers"
            :items="companyName"
            item-key="company"
            class="elevation-1"
            :search="search"
            v-model="selected"
            select-strategy="single"
            show-select
          >
            <template v-slot:top>
              <v-text-field
                v-model="search"
                label="Search Company Name"
                class="mx-4"
              ></v-text-field>
            </template>
    </v-data-table>
    </v-container> -->
  </v-app>
</template>

<script>
import Loading from '@/components/Loading'
import comapnyTabel from '@/components/companyTable'
export default {
  name: 'Collaboration',
  components: {
    Loading,
    comapnyTabel
  },
  data () {
    return {
      // companyName: [],
      // search: '',
      // selected: [],
      // // predictedTerm: [],
      // t1: [],
      // t2: [],
      // t3: [],
      // t4: [],
      // isShow: false,
      // dialog: false
    }
  },
  computed: {
    // headers () {
    //   return [
    //     {
    //       text: '会社名',
    //       align: 'start',
    //       sortable: false,
    //       value: 'company'
    //     }
    //   ]
    // }
  },
  methods: {
    // async showPrediction () {
    //   console.log(this.selected)
    //   const path = process.env.VUE_APP_BASE_URL + 'api/predict'
    //   const data = []
    //   for (let i = 0; i < this.selected.length; i++) {
    //     data.push(this.selected[i].company)
    //   }
    //   const params = {
    //     company: data
    //   }
    //   console.log(params)
    //   this.isShow = true
    //   await this.$api
    //     .post(path, params)
    //     .then(response => {
    //       console.log(response.data)
    //       this.$router.push({
    //         name: 'prediction',
    //         params: { responseData: response.data }
    //       })
    //       // this.predictedTerm = response.data.recommendable_items
    //       this.t1 = response.data.t1
    //       this.t2 = response.data.t2
    //       this.t3 = response.data.t3
    //       this.t4 = response.data.t4
    //       // selectedを初期化
    //       this.selected.splice(0, this.selected.length)
    //       this.isShow = false
    //     })
    //     .catch(error => {
    //       console.log(error)
    //     })
    // }
  }
  // async created () {
  //   const path = process.env.VUE_APP_BASE_URL + 'api/getCompanyName'
  //   await this.$api
  //     .post(path)
  //     .then(response => {
  //       this.companyName = response.data.companyList
  //     })
  //     .catch(error => {
  //       console.log(error)
  //     })
  //   console.log(this.companyName)
  // }
}

</script>

<style scoped>

</style>
