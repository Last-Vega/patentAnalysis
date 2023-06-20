<template>
  <v-app>
  <Loading :flag="isShow" />
  <v-container>
    熊谷組とコラボする企業を選択してください
    <!-- <v-btn
    color="primary"
    @click="showPrediction"
    >
    推薦結果を見る
    </v-btn> -->
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

      <v-card>
        <v-card-title class="text-h5 grey lighten-2">
          推薦結果
        </v-card-title>
        <v-simple-table>
          <template v-slot:default>
            <caption>
              単語
            </caption>
            <thead>
              <tr>
                <th class="text-left">Caption</th>
                <th class="text-left">Term</th>
              </tr>
            </thead>
            <tbody>
              <!-- <tr
                v-for="(term, index) in predictedTerm"
                :key="index"
                >
                <td>{{ term }}</td>
              </tr> -->
              <tr
                v-for="(term, index) in t1.term"
                :key="index"
                :style="{'background-color': t1.color}"
                >
                <td>◯(熊谷組)(他社)</td>
                <td>{{ term }}</td>
              </tr>
              <tr
                v-for="(term, index) in t2.term"
                :key="index"
                :style="{'background-color': t2.color}"
                >
                <td>熊谷組のみ保有</td>
                <td>{{ term }}</td>
              </tr>
              <tr
                v-for="(term, index) in t3.term"
                :key="index"
                :style="{'background-color': t3.color}"
                >
                <td>他社のみ保有</td>
                <td>{{ term }}</td>
              </tr>
              <tr
                v-for="(term, index) in t4.term"
                :key="index"
                :style="{'background-color': t4.color}"
                >
                <td>×(熊谷組)(他社)</td>
                <td>{{ term }}</td>
              </tr>
            </tbody>
          </template>
        </v-simple-table>
        <v-divider></v-divider>

        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn
            color="primary"
            text
            @click="dialog = false"
          >
            閉じる
          </v-btn>
        </v-card-actions>
      </v-card>
</v-dialog>
  <v-data-table
          :headers="headers"
          :items="companyName"
          item-key="company"
          class="elevation-1"
          :search="search"
          v-model="selected"
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
</v-container>
</v-app>
</template>

<script>
import Loading from '@/components/Loading'
export default {
  name: 'Collaboration',
  components: {
    Loading
  },
  data () {
    return {
      companyName: [],
      search: '',
      selected: [],
      // predictedTerm: [],
      t1: [],
      t2: [],
      t3: [],
      t4: [],
      isShow: false,
      dialog: false
    }
  },
  computed: {
    headers () {
      return [
        {
          text: '会社名',
          align: 'start',
          sortable: false,
          value: 'company'
        }
      ]
    }
  },
  methods: {
    async showPrediction () {
      console.log(this.selected)
      const path = process.env.VUE_APP_BASE_URL + 'api/predict'
      const data = []
      for (let i = 0; i < this.selected.length; i++) {
        data.push(this.selected[i].company)
      }
      const params = {
        company: data
      }
      console.log(params)
      this.isShow = true
      await this.$api
        .post(path, params)
        .then(response => {
          console.log(response.data)
          // this.$router.push({
          //   name: 'Prediction',
          //   params: { prediction: response.data }
          // })
          // this.predictedTerm = response.data.recommendable_items
          this.t1 = response.data.t1
          this.t2 = response.data.t2
          this.t3 = response.data.t3
          this.t4 = response.data.t4
          // selectedを初期化
          this.selected.splice(0, this.selected.length)
          this.isShow = false
        })
        .catch(error => {
          console.log(error)
        })
    }
  },
  async created () {
    const path = process.env.VUE_APP_BASE_URL + 'api/getCompanyName'
    await this.$api
      .post(path)
      .then(response => {
        this.companyName = response.data.companyList
      })
      .catch(error => {
        console.log(error)
      })
    console.log(this.companyName)
  }
}

</script>

<style scoped>

</style>
