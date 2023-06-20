<template>
<div>
  <v-row justify="center">
    <v-dialog
      v-model="dialog"
      fullscreen
      hide-overlay
      transition="dialog-bottom-transition"
    >
      <template v-slot:activator="{ on, attrs }">
        <v-btn
          color="primary"
          dark
          v-bind="attrs"
          v-on="on"
        >
          推薦結果を表示する
        </v-btn>
      </template>
      <v-card class="card">
        <Loading :flag="isShow" />
        <v-toolbar
          dark
          color="primary"
        >
          <v-btn
            icon
            dark
            @click="dialog = false"
          >
            <v-icon>mdi-close</v-icon>
          </v-btn>
          <v-toolbar-title>推薦結果</v-toolbar-title>
          <v-spacer></v-spacer>
          <v-toolbar-items>
          </v-toolbar-items>
        </v-toolbar>

          <v-subheader>熊谷組におすすめの技術用語</v-subheader>
          <v-btn
          dark
          text
          color="primary"
          @click="submit"
        >
          推薦件結果を見る
        </v-btn>

        <v-divider></v-divider>
        {{length}}個の技術用語を表示しています．
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
                v-for="(term, index) in closeTerm"
                :key="index"
                >
                <td>{{term}}</td>
              </tr>
            </tbody>
          </template>
        </v-simple-table>
        <!-- <v-list
          three-line
          subheader
        >
          <v-subheader>General</v-subheader>
          <v-list-item>
            <v-list-item-action>
              <v-checkbox v-model="notifications"></v-checkbox>
            </v-list-item-action>
            <v-list-item-content>
              <v-list-item-title>Notifications</v-list-item-title>
              <v-list-item-subtitle>Notify me about updates to apps or games that I downloaded</v-list-item-subtitle>
            </v-list-item-content>
          </v-list-item>
          <v-list-item>
            <v-list-item-action>
              <v-checkbox v-model="sound"></v-checkbox>
            </v-list-item-action>
            <v-list-item-content>
              <v-list-item-title>Sound</v-list-item-title>
              <v-list-item-subtitle>Auto-update apps at any time. Data charges may apply</v-list-item-subtitle>
            </v-list-item-content>
          </v-list-item>
          <v-list-item>
            <v-list-item-action>
              <v-checkbox v-model="widgets"></v-checkbox>
            </v-list-item-action>
            <v-list-item-content>
              <v-list-item-title>Auto-add widgets</v-list-item-title>
              <v-list-item-subtitle>Automatically add home screen widgets</v-list-item-subtitle>
            </v-list-item-content>
          </v-list-item>
        </v-list> -->
      </v-card>
    </v-dialog>
  </v-row>
</div>
</template>

<script>
import Loading from '@/components/Loading'
export default {
  components: {
    Loading
  },
  data () {
    return {
      dialog: false,
      notifications: false,
      sound: true,
      widgets: false,
      closeTerm: [],
      isShow: false,
      length: 0
    }
  },
  methods: {
    async submit () {
      this.isShow = true
      const path = process.env.VUE_APP_BASE_URL + 'api/recommend'
      await this.$api
        .post(path)
        .then(response => {
          this.closeTerm.splice(0, this.closeTerm.length)
          this.closeTerm = response.data.closeTerm
          this.length = response.data.length
          this.isShow = false
          // alert(response.data.message)
        })
        .catch(error => {
          console.log(error)
          this.isShow = false
        })
    }
  }
}
</script>

<style scoped>
.card {
  z-index: 1;
}
</style>
