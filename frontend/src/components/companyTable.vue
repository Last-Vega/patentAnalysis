<template>
  <v-app>
    <Loading :flag="isShow" />
    <v-container>
      <v-autocomplete
              ref="targetCompany"
              v-model="targetCompany"
              :rules="[() => !!targetCompany || 'Company is required']"
              :items="companyName"
              placeholder="株式会社熊谷組"
              required
            ></v-autocomplete>
      に
      <v-autocomplete
        v-model="selected"
        :items="companyName"
        chips
        hide-details
        hide-no-data
        hide-selected
        multiple
        single-line
      ></v-autocomplete>
      <v-btn
          dark
          v-on="on"
          @click="showPrediction"
        >
      を近づける
      </v-btn>
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
      targetCompany: null,
      search: '',
      selected: [],
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
      const path = process.env.VUE_APP_BASE_URL + 'api/predict'
      const params = {
        targetCompany: this.targetCompany,
        collaboratedCompany: this.selected
      }
      this.isShow = true

      await this.$api
        .post(path, params)
        .then(response => {
          console.log(response.data)
          this.$router.push({
            name: 'prediction',
            params: { responseData: response.data }
          })
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
    const path = process.env.VUE_APP_BASE_URL + 'api/getCompanyNameList'
    await this.$api
      .post(path)
      .then(response => {
        this.companyName = response.data.companyList
      })
      .catch(error => {
        console.log(error)
      })
    // console.log(this.companyName)
  }
}

</script>

<style scoped></style>
