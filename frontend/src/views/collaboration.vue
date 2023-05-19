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
                <th class="text-left">Term</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(term, index) in predictedTerm"
                :key="index"
                >
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
      predictedTerm: [],
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
      this.isShow = true
      await this.$api
        .post(path, params)
        .then(response => {
          console.log(response.data)
          // this.$router.push({
          //   name: 'Prediction',
          //   params: { prediction: response.data }
          // })
          this.predictedTerm = response.data.recommendable_items
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
