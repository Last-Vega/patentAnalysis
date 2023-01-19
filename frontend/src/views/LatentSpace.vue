<template>
  <v-app>
    <Loading :flag="isShow" />
    <v-row no-gutters>
      <v-col cols="12" sm="8">
        <div v-if="this.updateCompany.length > 0 || this.updateTerm.length > 0">
            <div class="text-center">
                <v-btn color="red lighten-2" dark @click="updateZ">
                  更新する
                </v-btn>
            </div>
          </div>
        <ViewLatentSpace
          :options="options"
          :companyItems="CCContrib"
          :termItems="CTContrib"
          @toggle="toggle"
          @change-label="changeLabel"
          />
      </v-col>

      <v-col cols="12" sm="4">

        <!-- データの座標一覧 -->
        <v-simple-table dense fixed-header height="300px">
          <template v-slot:default>
            <caption>
              要素の座標
            </caption>
            <thead>
              <tr>
                <th class="text-left">name</th>
                <th class="text-left">X</th>
                <th class="text-left">Y</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in companyXY" :key="item.dataIndex">
                <td>{{ item.label }}</td>
                <td class="text-right">
                  <v-text-field
                    type="number"
                    v-model.number="item.x"
                    @change="addUpdateIndex('company', item.dataIndex)"
                  />
                </td>
                <td class="text-right">
                  <v-text-field
                    type="number"
                    v-model.number="item.y"
                    @change="addUpdateIndex('company', item.dataIndex)"
                  />
                </td>
              </tr>
            </tbody>
          </template>
        </v-simple-table>

        <hr style="margin: 1rem 0;">

        <!-- 指定場所に移動 -->
        <v-simple-table dense fixed-header>
          <template v-slot:default>
            <caption>
              指定された場所に要素を移動する
            </caption>
            <tbody>
              <tr>
                <td>
                  移動する要素
                </td>
                <td>
                  <v-autocomplete
                    v-model="fromPointName"
                    :items="companyName.concat(termName)"
                    dense
                    filled
                  ></v-autocomplete>
                </td>
                <td></td>
              </tr>
              <tr>
                <td>
                  移動先の要素
                </td>
                <td>
                <v-autocomplete
                  v-model="toPointName"
                  :items="companyName.concat(termName)"
                  dense
                  filled
                ></v-autocomplete>
                </td>
                <td>
                  <v-btn v-on:click="movePoint">
                    移動
                  </v-btn>
                </td>
              </tr>
            </tbody>
          </template>
        </v-simple-table>

        <hr style="margin: 1rem 0;">

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
      isDraggable: true,
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
      CTContrib: '',
      fromPointName: '',
      toPointName: '',
      history: []
    }
  },
  methods: {
    addUpdateIndex (type, idx) {
      if (type === 'company') {
        if (!this.updateCompany.includes(idx)) {
          this.updateCompany.push(idx)
        }
      } else if (type === 'term') {
        if (!this.updateTerm.includes(idx)) {
          this.updateTerm.push(idx)
        }
      }
    },
    movePoint () {
      // 選択された要素のindex取得
      const concatName = this.companyName.concat(this.termName)
      const toIdx = concatName.indexOf(this.toPointName)
      const fromIdx = concatName.indexOf(this.fromPointName)
      if (toIdx === -1 || fromIdx === -1) {
        return
      }

      // 移動先の座標取得
      const { x, y } = ((toIdx) => {
        if (toIdx < this.companyName.length) {
          const { x, y } = this.companyXY[toIdx]
          return { x, y }
        } else {
          const { x, y } = this.termXY[toIdx - this.companyName.length]
          return { x, y }
        }
      })(toIdx)

      // 移動させるデータを取得（ + updateIndex に追加）
      const fromData = ((fromIdx) => {
        if (fromIdx < this.companyName.length) {
          this.addUpdateIndex('company', fromIdx)
          return this.companyXY[fromIdx]
        } else {
          const idx = fromIdx - this.companyName.length
          this.addUpdateIndex('term', idx)
          return this.termXY[idx]
        }
      })(fromIdx)

      // データの座標を設定
      fromData.x = x
      fromData.y = y
    },
    changeLabel (check) {
      const end = check ? 100 : 3
      this.companyXY.forEach((v, i) => {
        v.company = this.labelFormat(this.companyName[i], end)
      })
      this.termXY.forEach((v, i) => {
        v.term = this.labelFormat(this.termName[i], end)
      })
    },
    toggle () {
      this.isDraggable = !this.isDraggable
      this.options.series[0].dragDrop = {
        draggableX: this.isDraggable,
        draggableY: this.isDraggable,
        liveRedraw: this.isDraggable
      }
      this.options.series[1].dragDrop = {
        draggableX: this.isDraggable,
        draggableY: this.isDraggable,
        liveRedraw: this.isDraggable
      }
    },
    labelFormat (s, end = 3) {
      return s.replace('株式会社', '').slice(0, end)
    },
    makeScatter (company, term) {
      console.log(company)
      this.companyXY = company.map((v, i) => {
        return {
          dataIndex: i,
          label: this.companyName[i],
          x: v[0],
          y: v[1],
          company: this.labelFormat(this.companyName[i])
        }
      })
      this.termXY = term.map((v, i) => {
        return {
          dataIndex: i,
          label: this.termName[i],
          x: v[0],
          y: v[1],
          term: this.labelFormat(this.termName[i])
        }
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

      const path = process.env.VUE_APP_BASE_URL + 'api/update'
      const postData = {
        companyZ: this.options.series[0].data.map(v => [v.x, v.y]),
        termZ: this.options.series[1].data.map(v => [v.x, v.y]),
        CompanyIndex: this.updateCompany,
        TermIndex: this.updateTerm
      }
      // console.log(postData)

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
    },
    prep (companyData, termData) {
      for (let i = 0; i < companyData.length; i++) {
        this.companyName.push(companyData[i].company)
        this.companyXY.push({
          dataIndex: i,
          x: companyData[i].x,
          y: companyData[i].y,
          label: companyData[i].company,
          company: this.labelFormat(companyData[i].company)
        })
      }
      for (let i = 0; i < termData.length; i++) {
        this.termName.push(termData[i].term)
        this.termXY.push({
          dataIndex: i,
          x: termData[i].x,
          y: termData[i].y,
          label: termData[i].term,
          term: this.labelFormat(termData[i].term)
        })
      }
      // console.log(this.companyXY)
      this.options.series[0].dataLabal = this.companyName
      this.options.series[0].data = this.companyXY
      this.options.series[1].dataLabal = this.termName
      this.options.series[1].data = this.termXY
    }
  },
  async created () {
    const path = process.env.VUE_APP_BASE_URL + 'api/latent'
    await this.$api
      .post(path)
      .then(response => {
        if (response.data.flag === true) {
          const companyData = response.data.companyInfo.key
          const termData = response.data.termInfo.key
          this.prep(companyData, termData)
        } else {
          const companyData = companyInfo.key
          const termData = termInfo.key
          this.prep(companyData, termData)
        }
      })
      .catch(error => {
        console.log(error)
        this.isShow = false
      })

    // const companyData = companyInfo.key
    // const termData = termInfo.key
    // for (let i = 0; i < companyData.length; i++) {
    //   this.companyName.push(companyData[i].company)
    //   this.companyXY.push({
    //     dataIndex: i,
    //     x: companyData[i].x,
    //     y: companyData[i].y,
    //     label: companyData[i].company,
    //     company: this.labelFormat(companyData[i].company)
    //   })
    // }
    // for (let i = 0; i < termData.length; i++) {
    //   this.termName.push(termData[i].term)
    //   this.termXY.push({
    //     dataIndex: i,
    //     x: termData[i].x,
    //     y: termData[i].y,
    //     label: termData[i].term,
    //     term: this.labelFormat(termData[i].term)
    //   })
    // }
    // // console.log(this.companyXY)
    // this.options.series[0].dataLabal = this.companyName
    // this.options.series[0].data = this.companyXY
    // this.options.series[1].dataLabal = this.termName
    // this.options.series[1].data = this.termXY
  }
}
</script>
