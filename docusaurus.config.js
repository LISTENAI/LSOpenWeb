module.exports = {
  title: '聆思文档中心',
  tagline: 'LSOpen资料库',
  url: 'https://github.com/LISTENAI',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'listenai', // Usually your GitHub org/user name.
  projectName: 'LSOpenWeb', // Usually your repo name.
  organizationName: 'LISTENAI',
  themeConfig: {
    navbar: {
      title: '聆思文档中心',
      logo: {
        href: '/',
        alt: 'LSOpen Logo',
        src: 'img/logo.png',
      },
      items: [
        // {
        //   to: "/",
        //   label: 'CSK开发指南',
        //   position: 'left'
        // },
        {
          to: "/CSK_online_guides_home",
          label: '离在线开发指南',
          position: 'left'
        },
        {
          to: "/csksdk_api",
          label: 'API',
          position: 'left'
        },
        {
          to: '/download',
          label: '下载中心',
          position: 'left'
        },
        {
          to: '/lisa',
          label: 'Hello Lisa',
          position: 'left'
        },
        {
          to: '/school',
          label: '视频教程',
          position: 'left'
        },
        {
          href: "https://cloud.listenai.com",
          label: "LSCloud",
          position: 'right'
        }
      ],
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false
    },
    footer: {
      style: 'dark',
      links: [],
      copyright: `Copyright © 安徽聆思智能科技有限公司皖ICP备05001217号`,
    }
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: '/'
          // Please change this to your repo.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/edit/master/website/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  plugins: [
    // ... Your other plugins.
    'lstenai-analytics',
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        // ... Your options.
        // `hashed` is recommended as long-term-cache of index file is possible.
        hashed: true,
        indexPages: true,
        docsRouteBasePath: "/",
        // For Docs using Chinese, The `language` is recommended to set to:
        // ```
        language: ["en", "zh"],
        // ```
        // When applying `zh` in language, please install `nodejieba` in your project.
      },
    ],
  ],
};
