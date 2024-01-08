// index.js
Page({
  imageTap1:function(){
      wx.navigateTo({
          url: '/pages/talk/talk',
      })
  },
  imageTap:function(){
    wx.navigateTo({
        url: '/pages/my/my',
    })
  }
}
)