<template>
  <v-app>
    <Loading :flag="isShow" />
    <v-row no-gutters>
      <v-col cols="12" sm="9">
        <div v-if="this.updateCompany.length >= 0 || this.updateTerm.length > 0">
            <div class="text-center">
                <v-btn color="red lighten-2" dark @click="updateZ">
                  更新する
                </v-btn>
            </div>
          </div>
        <ViewLatentSpace :options="options" :companyItems="CCContrib" :termItems="CTContrib" />
      </v-col>

      <v-col cols="12" sm="3">
        <ViewTabel :companyName="companyName" :termName="termName" :companyZ="this.options.series[0].data" :termZ="this.options.series[1].data" />
      </v-col>
    </v-row>
  </v-app>
</template>

<script>
import {
  companyTableData,
  termTableData,
  chartOptions,
  updateCompanyIndex,
  updateTermIndex
} from '@/components/createLatentSpace'
// import companyInfo from '@/assets/latentC1223.json'
// import termInfo from '@/assets/latentT1223.json'
// import companyInfo from '@/assets/latentC0119.json'
// import termInfo from '@/assets/latentT0119.json'
import companyInfo from '@/assets/SwitchedlatentC0119.json'
import termInfo from '@/assets/SwitchedlatentT0119.json'
import ViewTabel from '@/components/viewTable'
import ViewLatentSpace from '@/components/viewLatentInfo'
import Loading from '@/components/Loading'

export default {
  name: 'LatentSpace',
  components: {
    ViewTabel,
    ViewLatentSpace,
    Loading
    // Recommendation
  },
  data () {
    return {
      options: chartOptions,
      companyItems: companyTableData,
      termItems: termTableData,
      companyName: [],
      companyXY: [],
      termName: [],
      termXY: [],
      query: '',
      updateCompany: updateCompanyIndex,
      updateTerm: updateTermIndex,
      isShow: false,
      maxCCPath: '',
      maxCTPath: '',
      CCContrib: '',
      CTContrib: ''
    }
  },
  methods: {
    labelFormat (s) {
      return s.replace('株式会社', '').slice(0, 3)
    },
    makeScatter (company, term) {
      console.log(company)
      this.companyXY = company.map((v, i) => {
        return { x: v[0], y: v[1], company: this.labelFormat(this.companyName[i]) }
      })
      this.termXY = term.map((v, i) => {
        return { x: v[0], y: v[1], term: this.labelFormat(this.termName[i]) }
      })
      this.options.series[0].data = this.companyXY
      this.options.series[1].data = this.termXY
    },
    interpretation (ccPath, ctPath) {
      const path = {
        C: '企業',
        T: '技術用語',
        Y: '公開年',
        I: '筆頭IPC',
        F: 'Fターム',
        P: '-特許-'
      }
      this.CTContrib = ctPath.split('').map(v => path[v]).join('')
      this.CCContrib = ccPath.split('').map(v => path[v]).join('')
      console.log(this.CCContrib)
    },
    async updateZ () {
      this.isShow = true
      console.log(this.updateComapny)
      console.log(this.updateTerm)

      // const rand = () => Math.random() * 6 - 3
      // const company = this.companyXY.map(v => [rand(), rand()])
      // const term = this.termXY.map(v => [rand(), rand()])
      // this.makeScatter(company, term)
      // this.isShow = false
      // this.interpretation('CPTPC', 'CPYPT')

      const path = process.env.VUE_APP_BASE_URL + 'api/update'
      const postData = {
        // companyZ: this.options.series[0].data,
        // termZ: this.options.series[1].data,
        companyZ: this.options.series[0].data.map(v => [v.x, v.y]),
        termZ: this.options.series[1].data.map(v => [v.x, v.y]),
        CompanyIndex: this.updateCompany,
        TermIndex: this.updateTerm
      }
      // console.log(postData)
      // console.log(postData.companyZ)
      // console.log(postData.termZ)
      await this.$api
        .post(path, postData)
        .then(response => {
          this.updateCompany.splice(0, this.updateCompany.length)
          this.updateTerm.splice(0, this.updateTerm.length)
          console.log(response.data.company)
          this.makeScatter(response.data.company, response.data.term)
          this.isShow = false
          this.interpretation(response.data.maxCCPath, response.data.maxCTPath)
        })
        .catch(error => {
          console.log(error)
          this.isShow = false
        })
    }
  },
  created () {
    const companyData = companyInfo.key
    const termData = termInfo.key
    for (let i = 0; i < companyData.length; i++) {
      this.companyName.push(companyData[i].company)
      this.companyXY.push({
        x: companyData[i].x,
        y: companyData[i].y,
        company: this.labelFormat(companyData[i].company)
      })
    }
    for (let i = 0; i < termData.length; i++) {
      this.termName.push(termData[i].term)
      this.termXY.push({
        x: termData[i].x,
        y: termData[i].y,
        term: this.labelFormat(termData[i].term)
      })
    }
    console.log(this.companyXY)
    this.options.series[0].dataLabal = this.companyName
    this.options.series[0].data = this.companyXY
    this.options.series[1].dataLabal = this.termName
    this.options.series[1].data = this.termXY
  }
}
</script>
