WRMCB=function(e){var c=console;if(c&&c.log&&c.error){c.log('Error running batched script.');c.error(e);}}
;
try {
/* module-key = 'com.atlassian.confluence.plugins.pagetree:pagetree-js-resources', location = 'soy/pagetree.soy' */
// This file was automatically generated from pagetree.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Pagetree.Macro.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Pagetree == 'undefined') { Confluence.Templates.Pagetree = {}; }
if (typeof Confluence.Templates.Pagetree.Macro == 'undefined') { Confluence.Templates.Pagetree.Macro = {}; }


Confluence.Templates.Pagetree.Macro.loadingIndicator = function(opt_data, opt_ignored) {
  return '<div class="plugin_pagetree_children_loading_wrapper"><div class="spinner"></div><span class="plugin_pagetree_children_loading">' + soy.$$escapeHtml('Loading...') + '</span></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Pagetree.Macro.loadingIndicator.soyTemplateName = 'Confluence.Templates.Pagetree.Macro.loadingIndicator';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.pagetree:pagetree-js-resources', location = 'com/atlassian/confluence/plugins/pagetree/resource/javascript/pagetree.js' */
(function(){var a=function(j){this.$=j;var f=this.$;var y=999;var z=true;var i=false;var u={};var c={};var k=function(I,E,J,D,B){var G=I.dataset.expanded==="true";if(E==undefined||E==null){E=!G}if(J==undefined||J==null){J=0}var F=I.getAttribute("data-children-loaded")==="true";if(G!==E){var C=I.getAttribute("data-page-id")+"-"+I.getAttribute("data-tree-id");var H=f("#children"+C);if(F){if(E==z){I.classList.remove("aui-iconfont-chevron-right");I.classList.add("aui-iconfont-chevron-down");I.setAttribute("data-expanded",true);I.setAttribute("aria-expanded",true)}else{I.classList.remove("aui-iconfont-chevron-down");I.classList.add("aui-iconfont-chevron-right");I.setAttribute("data-expanded",false);I.setAttribute("aria-expanded",false)}if(E==z){if(B==false){H.slideDown(300)}else{H.removeClass("plugin-pagetree-hide-children");H.animate({opacity:1})}}else{if(B==false){H.slideUp(300)}else{H.animate({opacity:0},{complete:function(){H.addClass("plugin-pagetree-hide-children")}})}}if(D){p(C)}}else{h(I,C,new Array(),J,"",D,B,true)}}if(D){p(C)}};var g=function(D,C,B){x(D,z,C,B)};var v=function(D,C,B){x(D,i,C,B)};var x=function(G,E,D,C){e(G);var F=f("#"+G);var B=F.find('a.plugin_pagetree_childtoggle[data-type="toggle"]');B.each(function(H){k(this,E,y,H==B.length-1,C)});D.preventDefault();D.stopPropagation()};var s=function(B){if(!B||B==undefined){return null}if(B.indexOf("plusminus")!=-1){return B.substring("plusminus".length)}if(B.indexOf("children")!=-1){return B.substring("children".length)}return null};var b=function(B){if(!B||B==undefined){return null}return o(B)[1]};var o=function(B){if(!B||B==undefined){return null}return B.split("-")};var e=function(C){var B=b(C);f("div.plugin_pagetree").each(function(D){if(D==B){f(this).find("span.plugin_pagetree_status").removeClass("hidden");f(this).find("div.plugin_pagetree_expandcollapse").addClass("hidden")}})};var p=function(C){var B=b(C);f("div.plugin_pagetree").each(function(D){if(D==B){f(this).find("span.plugin_pagetree_status").addClass("hidden");f(this).find("div.plugin_pagetree_expandcollapse").removeClass("hidden")}})};var d=function(G,E,L,I,H,D,C,J,F,K){var B=u[G];if(E=="Orphan"){B+="&hasRoot=false&spaceKey="+H}else{B+="&hasRoot=true&pageId="+E}B+="&treeId="+G+"&startDepth="+I+"&mobile="+C;f.each(L,function(){B+="&ancestors="+this});if(J){B+="&loadMore=true"}if(F!=null){B+="&elementsAfter="+F}if(K!=null){B+="&elementsBefore="+K}B+="&treePageId="+D;if(C==false){B=(AJS.params.serverUrl||"")+B}return B};var n=function(D){var C=D;var B=null;f("div.plugin_pagetree").each(function(E){if(E==C){B=f(this)}});return B};var m=function(B){var C=B.children("fieldset");var D=new Object();if(C.length>0){C.children("input").each(function(){D[this.name]=this.value})}return D};var l=function(B){var D=B.children("fieldset");var E=new Array();if(D.length>0){var C=D.children("fieldset");if(C.length>0){C.children("input").each(function(){E.push(this.value)})}}return E};var r=function(B){var C=f(document.createElement("div"));C.html(B);return C.find("ul[id^='child_ul']").length};var h=function(F,J,B,U,K,E,C,S,G,O,L,H){var Q=J;var I=E;var R=o(J);var M=R[0];var D=R[1];var V=f("#children"+J);var P=m(n(D));var T=false;function W(){if(O||L){F.innerHTML="Loading..."}else{var X=Confluence.Templates.Pagetree.Macro.loadingIndicator();V.find(".plugin_pagetree_children_loading_wrapper .spinner").spin("small");if(L){V.html(X+V.html())}else{V.append(X)}}}var N=0;if(typeof H!=="undefined"&&H){N=f("#"+H).offset().top}if(S){setTimeout(function(){if(!T){W()}},250)}else{W()}f.ajax({type:"GET",url:d(D,M,B,U,P.spaceKey,P.treePageId,C,G,O,L),dataType:"text",success:function(Y){if(r(Y)){f(".plugin_pagetree_children_loading_wrapper").remove();if(L!=null){var X=V.html();V.html(Y+X)}else{V.append(Y)}T=true;if(typeof H!=="undefined"){var Z=f("#"+H).offset().top-N;document.getElementsByClassName("acs-side-bar")[0].scrollTop=document.getElementsByClassName("acs-side-bar")[0].scrollTop+Z}if(V.children().length&&V.hasClass("hidden")){V.removeClass("hidden")}f("#plusminus"+Q).addClass("aui-iconfont-chevron-down").removeClass("aui-iconfont-chevron-right");f("#childrenspan"+c[parseInt(D)]+"-"+D).addClass("plugin_pagetree_current");if(I){p(Q)}if(typeof F!=="undefined"){F.setAttribute("data-children-loaded",true);F.setAttribute("data-expanded",true);F.setAttribute("aria-expanded",true)}if(O){f("#children"+J+" > ul > .page-tree-load-more-link-after").remove()}if(L){f("#children"+J+" > ul > .page-tree-load-more-link-before").remove()}q(V);if(AJS.PageGadget&&AJS.PageGadget.contentsUpdated){AJS.PageGadget.contentsUpdated()}}else{window.location=P.loginUrl}AJS.trigger("pagetree-children-loaded")},error:function(X){if(X.status=="403"){V.text(P["i18n-pagetree.error.permission"])}else{V.text(P["i18n-pagetree.error.general"])}}})};var q=function(B){f("div.plugin_pagetree_children_container:empty",B).addClass("hidden")};function t(F){var E=F.target;var D=E.getAttribute("data-type");var C=F.currentTarget.getAttribute("data-mobile")==="true";if(D==="toggle"){F.preventDefault();F.stopPropagation();k(F.target,null,null,null,C)}else{if(D==="load-all-after"){var I=E.getAttribute("data-last-rendered-id");var H=E.getAttribute("data-page-id")+"-"+E.getAttribute("data-tree-id");h(F.target,H,new Array(),null,"","",C,false,true,I,null)}else{if(D==="load-all-before"){var B=E.getAttribute("data-first-rendered-id");var H=E.getAttribute("data-page-id")+"-"+E.getAttribute("data-tree-id");var G="childrenspan"+B+"-"+E.getAttribute("data-tree-id");h(F.target,H,new Array(),null,"","",C,false,false,null,B,G)}else{return}}}F.preventDefault();F.stopPropagation()}var w=function(B,G,I){B[0].addEventListener("click",t,{passive:false});var E=m(B);var H=E.noRoot=="true";var D=H?"Orphan-"+G:E.rootPageId+"-"+G;var C=E.mobile=="true";if(C){B[0].setAttribute("data-mobile",C)}u[G]=E.treeRequestId;if(C==false){c[G]=AJS.params.pageId}else{c[G]=f("div.content-container").parent().attr("data-content-id")}B.children("fieldset").each(function(){var J=f(this);J.children("input[treeId]").attr("value",G);J.children("input[rootPage]").attr("value",D)});if(H){B.find("div.plugin_pagetree_children").attr("id","childrenOrphan-"+G);B.find("a.plugin_pagetree_expandall").click(function(J){g("childrenOrphan-"+G,J,C);return false});B.find("a.plugin_pagetree_collapseall").click(function(J){v("childrenOrphan-"+G,J,C);return false})}else{B.find("div.plugin_pagetree_children").attr("id","children"+D);B.find("a.plugin_pagetree_expandall").click(function(J){g("children"+D,J,C);return false});B.find("a.plugin_pagetree_collapseall").click(function(J){v("children"+D,J,C);return false})}var F=l(B);h(undefined,D,F,E.startDepth,E.spaceKey,"",C,false,I,null,null)};this.initPageTrees=function(B){f("div.plugin_pagetree").each(function(C){w(f(this),C,B)});A()};var A=function(){var B=self.placeFocus;if(B){self.placeFocus=function(){var C=f("form[name='pagetreesearchform']");C.attr("name","inlinecommentform");B();C.attr("name","pagetreesearchform")}}};f(".open-flyout-to-search").bind("click",function(B){if(f(".fly-out-open").length){setTimeout(function(){ConfluenceMobile.flyout.hide()},400)}else{setTimeout(function(){ConfluenceMobile.flyout.show()},400)}B.stopPropagation();B.preventDefault()})};Confluence=Confluence||{};Confluence.Plugins=Confluence.Plugins||{};Confluence.Plugins.PagetreeMacro={bind:function(b){var c=new a(b);c.initPageTrees(false)}}})();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.pagetree:pagetree-js-resources', location = 'com/atlassian/confluence/plugins/pagetree/resource/javascript/pagetree-desktop.js' */
AJS.toInit(function(a){Confluence.Plugins.PagetreeMacro.bind(a)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.extra.jira:help-dialog-extension', location = '/jira/help-dialog.js' */
Confluence.KeyboardShortcuts&&Confluence.KeyboardShortcuts.Editor.push({context:"editor.actions",descKey:"Insert Jira Issue/Filter"+":",keys:[["Ctrl+Shift+J"]]});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-mentions-plugin:help-dialog-extension', location = 'js/help-dialog.js' */
AJS.toInit(function($) {
    // Add the shortcut to the shortcuts dialog
    Confluence.KeyboardShortcuts && Confluence.KeyboardShortcuts.Autoformat.push(
        {
            action: "@",
            context: "autoformat.autocomplete",
            description: "Mention"
        });
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-dashboard:footer-analytics', location = '/js/confluence-dashboard/utils/footer-analytics.js' */
define("confluence-dashboard/utils/footer-analytics",["jquery","ajs","confluence/meta"],function(e,a,t){"use strict";var n=i(e).default,o=i(a).default,c=i(t).default;function i(e){return e&&e.__esModule?e:{default:e}}o.toInit(function(){var e=c.get("page-id"),a=n("body").hasClass("dashboard");n("#footer").on("click","a",function(){o.trigger("analytics",{name:"confluence.footer.item.click",data:{pageId:e,isDashboard:a}})})})}),require(["confluence-dashboard/utils/footer-analytics"]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.component.dialog2', location = 'aui.chunk.afb0455b22fe08f3d5f6--3bb4a4578cbbea02f437.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.component.dialog2"],{"0+IH":function(t,e,i){"use strict";i.r(e),i.d(e,"dialog2",(function(){return p}));i("AehQ"),i("rWoL");var a=i("+x/D"),u=i("TmQU"),n=i("KloK"),o=i("jEnx"),c=i("6RZY"),l=i("+ay7"),s=i("Pjwx"),d={"aui-focus":"false","aui-blanketed":"true"};function f(t){var e;this._handlers=new WeakMap,this.$el=t?Object(a.default)(t):Object(a.default)('\n        <section role="dialog" class="aui-layer aui-dialog2 aui-dialog2-medium">\n            <header class="aui-dialog2-header">\n                <h2 class="aui-dialog2-header-main"></h2>\n                '.concat(s.CLOSE_BUTTON,'\n                </button>\n            </header>\n            <div class="aui-dialog2-content"></div>\n            <footer class="aui-dialog2-footer"></footer>\n        </section>')),e=this.$el,a.default.each(d,(function(t,i){var a="data-"+t;e[0].hasAttribute(a)||e.attr(a,i)}))}f.prototype.on=function(t,e){const i=this.$el;if(!this._handlers.get(e)){const a=function(t){t.target===i.get(0)&&e.apply(this,arguments)};Object(o.default)(i).on(t,a),this._handlers.set(e,a)}return this},f.prototype.off=function(t,e){const i=this.$el,a=this._handlers.get(e);return a&&(Object(o.default)(i).off(t,a),this._handlers.delete(e)),this},f.prototype.show=function(){return Object(o.default)(this.$el).show(),this},f.prototype.hide=function(){return Object(o.default)(this.$el).hide(),this},f.prototype.remove=function(){return Object(o.default)(this.$el).remove(),this},f.prototype.isVisible=function(){return Object(o.default)(this.$el).isVisible()};var h=Object(c.default)("dialog2",f),r=new Set;h.on=function(t,e){return r.has(e)||(o.default.on(t,".aui-dialog2",e),r.add(e)),this},h.off=function(t,e){return r.has(e)&&(o.default.off(t,".aui-dialog2",e),r.delete(e)),this},Object(a.default)(document).on("click keydown",".aui-dialog2-header ".concat(s.CLOSE_BUTTON_CLASS_SELECTOR),(function(t){("click"===t.type||t.keyCode===l.default.ENTER||t.keyCode===l.default.SPACE)&&(t.preventDefault(),h(Object(a.default)(t.target).closest(".aui-dialog2")).hide())})),h.on("show",(function(t,e){e.find(s.CLOSE_BUTTON_CLASS_SELECTOR).attr("tabindex",0),e.attr("tabindex",-1);var i=e[0].hasAttribute("data-aui-focus-selector")&&e.attr("data-aui-focus-selector");if(i){var a=e.find(i);if(a.length>0)return a.first().focus()}e.focus()})),h.on("hide",(function(t,e){var i=Object(o.default)(e);e.data("aui-remove-on-hide")&&i.remove()})),Object(u.default)("aui/dialog2",h),Object(n.default)("dialog2",h);var p=h}},[["0+IH","runtime","aui.splitchunk.vendors--894c8113d9","aui.splitchunk.0d131bcbf1","aui.splitchunk.fbbef27525","aui.splitchunk.444efc83be","aui.splitchunk.739b9ec8cc","aui.splitchunk.dd803a46b4","aui.splitchunk.994e478d48","aui.splitchunk.d7c46c2734","aui.splitchunk.e54c7c7304","aui.splitchunk.fb15cffa72","aui.splitchunk.f1e06f97a4","aui.splitchunk.479fe6ee76","aui.splitchunk.f673ef53ac","aui.splitchunk.8659b532c1","aui.splitchunk.5f851f97df","aui.splitchunk.d0110a864f","aui.splitchunk.afa5039e04","aui.splitchunk.bff3715233","aui.splitchunk.c750721820","aui.splitchunk.6d6f245ed3","aui.splitchunk.5b8c290363","aui.splitchunk.3d2cb1af14"]]]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:page-loading-indicator', location = '/includes/js/page-loading-indicator.js' */
define("confluence/page-loading-indicator",["jquery","underscore","ajs","confluence/templates"],function(c,g,a,h){return function(b){function d(){return c(".confluence-page-loading-blanket",b)}function e(){return c(".confluence-loading-indicator",b)}return{show:function(){0===d().length&&c(b).append(h.pageLoadingIndicator());d().show();e().spin({lines:12,length:8,width:4,radius:10,trail:60,speed:1.5,color:"#F0F0F0"})},hide:function(){e().stop();d().hide()},showUntilResolved:function(c,b){this.show();
c.always(g.bind(this.hide,this));b&&c.fail(function(){a.messages.error(".confluence-page-loading-errors",{body:b})})},showUntilDialogVisible:function(b,d){var f=this,e=d||"An error has occurred connecting to the server. Please try again.",g=c(".aui-dialog:visible"),h=c(".aui-dialog2:visible");!g.length&&!h.length&&f.show();b.always(f.hide).fail(function(){a.messages.error(".confluence-page-loading-errors",{body:e})});a.bind("show.dialog",function i(){a.unbind("show.dialog",i);f.hide()});if(null!=a.dialog2&&void 0!=a.dialog2)a.dialog2.on("show",
function j(){a.dialog2.off("show",j);f.hide()})},destroy:function(){b.remove(".confluence-page-loading-blanket")}}}});require("confluence/module-exporter").exportModuleAsGlobal("confluence/page-loading-indicator","Confluence.PageLoadingIndicator");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:page-loading-indicator', location = '/includes/soy/page-loading-indicator.soy' */
// This file was automatically generated from page-loading-indicator.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }


Confluence.Templates.pageLoadingIndicator = function(opt_data, opt_ignored) {
  return '<div class="confluence-page-loading-errors"></div><div class="confluence-page-loading-blanket aui-blanket" aria-hidden="false"><div class="confluence-loading-indicator"></div></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.pageLoadingIndicator.soyTemplateName = 'Confluence.Templates.pageLoadingIndicator';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:deferred-loaders', location = '/includes/js/deferred-dialog-loader.js' */
define("confluence/deferred-dialog-loader",["underscore","jquery","ajs","confluence/page-loading-indicator","confluence/api/event"],function(c,d,i,j,b){return function(){var k=j(d("body")),e,g=!1,h=!1,f=function(){g&&h&&e&&(b.trigger(e),e=void 0)};b.bind("page.move.dialog.ready",function(){g=!0;f()});b.bind("blogpost.move.dialog.ready",function(){h=!0;f()});c.each({movePageDialogTools:{resource:"confluence.web.resources:page-move-resources",selector:"#action-move-page-dialog-link",event:"deferred.page-move.tools-menu"},
movePageDialogEditor:{resource:"confluence.web.resources:page-move-resources",selector:"#rte-button-location",event:"deferred.page-move.editor"},moveBlogDialogTools:{resource:"confluence.web.resources:page-move-resources",selector:"#action-move-blogpost-dialog-link",event:"deferred.blog-move.tools-menu"},availableGadgetsHelp:{resource:"com.atlassian.confluence.plugins.gadgets:gadget-directory-resources",selector:"#gadget-directory-link",event:"deferred.available-gadgets.help-menu"},aboutConfluenceHelp:{resource:"confluence.web.resources:about",
selector:"#confluence-about-link",event:"deferred.about-confluence.help-menu"}},function(a){d("body").on("click",a.selector,function(c){var d=a.resource+".requested",f=WRM.require("wr!"+a.resource,function(){"confluence.web.resources:page-move-resources"!==a.resource?b.trigger(a.event):g&&h?b.trigger(a.event):e=a.event});k.showUntilDialogVisible(f);i.Analytics?i.Analytics.triggerPrivacyPolicySafeEvent(d):b.trigger("analyticsEvent",{name:d});c.preventDefault()})})}});
require("confluence/module-exporter").safeRequire("confluence/deferred-dialog-loader",function(c){require("ajs").toInit(c)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:deferred-loaders', location = '/includes/js/page-permissions-deferred-loader.js' */
define("confluence/page-permissions-deferred-loader","confluence/dark-features confluence/legacy ajs confluence/page-loading-indicator jquery wrm".split(" "),function(a,b,c,d,e,f){var g=d(e("body"));return function(a){var b=f.require("wr!com.atlassian.confluence.plugins.confluence-page-restrictions-dialog:dialog-resources",function(){c.trigger("deferred.page.permissions")});g.showUntilDialogVisible(b,"There was an error loading the page restrictions. Please try again later.");a.preventDefault()}});
require("confluence/module-exporter").safeRequire("confluence/page-permissions-deferred-loader",function(a){var b=require("ajs");b.toInit(function(c){c("body").on("click","#rte-button-restrictions,#action-page-permissions-link",a);b.bind("system-content-metadata.open-restrictions-dialog",a)})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:navigator-context', location = '/includes/js/api/navigator-context.js' */
define("confluence/api/navigator-context",["confluence/meta","document"],function(g,b){var h=function(a){a=d(a);return"undefined"!==typeof a&&0!==a},d=function(a){a=g.get(a);if(!isNaN(a))return Number(a)},i=function(){return!!b.querySelector(".page.view")||!!b.querySelector(".blogpost.view")},e=function(){return!!b.querySelector(".page.edit")||!!b.querySelector(".blogpost.edit")},f=function(){return g.get("content-type")},c=function(){return d("page-id")};return{getCurrent:function(){return e()&&
0===c()&&h("draft-id")?{target:"contentcreate",context:{contentId:d("draft-id"),contentType:f()}}:e()&&!i()&&0!==c()?{target:"contentedit",context:{contentId:c(),contentType:f()}}:!e()&&i()&&h("page-id")?{target:"contentview",context:{contentId:c(),contentType:f()}}:{target:"unknown",context:{}}}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-hipchat-integration-plugin:hipchat-presence-resources', location = 'presence/presence.soy' */
// This file was automatically generated from presence.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.HipChat.Presence.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.HipChat == 'undefined') { Confluence.Templates.HipChat = {}; }
if (typeof Confluence.Templates.HipChat.Presence == 'undefined') { Confluence.Templates.HipChat.Presence = {}; }


Confluence.Templates.HipChat.Presence.badge = function(opt_data, opt_ignored) {
  return '<span class="hipchat-status aui-icon aui-icon-small hipchat-icon-' + soy.$$escapeHtml(opt_data.statusCode) + '" data-status="' + soy.$$escapeHtml(opt_data.statusCode) + '" title="' + soy.$$escapeHtml(opt_data.statusMessage) + '">' + soy.$$escapeHtml(opt_data.statusMessage) + '</span>';
};
if (goog.DEBUG) {
  Confluence.Templates.HipChat.Presence.badge.soyTemplateName = 'Confluence.Templates.HipChat.Presence.badge';
}


Confluence.Templates.HipChat.Presence.chatBar = function(opt_data, opt_ignored) {
  return '<div class="hipchat-chatbar aui-buttons"><button class="aui-button hipchat-chat" title="' + soy.$$escapeHtml('Unable to start chat as this user is not known in Hipchat.') + '" aria-disabled="true"><span><span class="aui-icon aui-icon-small hipchat-icon-chat">' + soy.$$escapeHtml('Chat') + '</span></span></button><button class="aui-button hipchat-voice" title="' + soy.$$escapeHtml('Unable to start voice calls as this user is not known in Hipchat.') + '" aria-disabled="true" data-call-type="voice"><span><span class="aui-icon aui-icon-small hipchat-icon-voice">' + soy.$$escapeHtml('Voice') + '</span></span></button><button class="aui-button hipchat-video" title="' + soy.$$escapeHtml('Unable to start video calls as this user is not known in Hipchat.') + '" aria-disabled="true" data-call-type="video"><span><span class="aui-icon aui-icon-small hipchat-icon-video">' + soy.$$escapeHtml('Video') + '</span></span></button></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.HipChat.Presence.chatBar.soyTemplateName = 'Confluence.Templates.HipChat.Presence.chatBar';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-hipchat-integration-plugin:hipchat-presence-resources', location = 'presence/presence.js' */
(function(d){var f=AJS.DarkFeatures.isEnabled("hipchat.chatbar");var a={hipChatCall:function(g,h){if(!g){return}return"hipchat://hipchat.com/user/"+encodeURI(g)+(h?"?call="+encodeURI(h):"")}};var c=AJS.contextPath()+"/rest/hipchat/integration/1.0/users/";var b=function(g){return function(j){var i=d(this);var h=i.attr("data-user-id");var k=i.attr("data-caller-id");if(!k){return}var l=i.attr("data-call-type");if(!g&&l){return}window.location.replace(a.hipChatCall(k,l));AJS.trigger("analyticsEvent",{name:"hipchat.chatbar."+(l||"chat"),data:{userId:h}});j.preventDefault()}};var e=function(g){return function(i){if(!i||!i.presence){return}var j=i.userId;var h=i.hipChatUserId;var k;var m;var l=d(g).find(".hipchat-chatbar .aui-button");l.attr("data-user-id",j);l.attr("data-caller-id",h);l.click(b(i.presence.is_online));if(i.presence.is_online){if(i.presence.show==="xa"||i.presence.show==="away"){k="away";m="Away"}else{if(i.presence.show==="dnd"){k="dnd";m="Do not disturb"}else{k="available";m="Available"}}l.filter(".hipchat-chat").attr("title","Start chat with this user in Hipchat.").removeAttr("aria-disabled");l.filter(".hipchat-voice").attr("title","Start a voice call with this user in Hipchat.").removeAttr("aria-disabled");l.filter(".hipchat-video").attr("title","Start a video call with this user in Hipchat.").removeAttr("aria-disabled")}else{k="offline";m="Offline";l.filter(".hipchat-chat").attr("title","Start chat with this user in Hipchat.").removeAttr("aria-disabled");l.filter(".hipchat-voice").attr("title","Unable to start voice calls as this user is not known in Hipchat.");l.filter(".hipchat-video").attr("title","Unable to start video calls as this user is not known in Hipchat.")}if(i.presence.status){m+=" ("+i.presence.status+")"}d("div.values",g).append(Confluence.Templates.HipChat.Presence.badge({statusCode:k,statusMessage:m}));AJS.trigger("analyticsEvent",{name:"hipchat.presence.user.found",data:{userId:i.userId}})}};d(document).bind("ajaxComplete",function(i,g,h){if(/userinfopopup\.action/.test(h.url)){d.each(d(".vcard"),function(n,p){var m=d(p);var k=m.closest(".contents");var l=k.find(".profile-macro").first();var o=k.find(".actions").first();if(l.hasClass("hipchat-status-applied")||!o.length){return}l.addClass("hipchat-status-applied");if(f){o.addClass("hipchat-chatbar-support");o.append(Confluence.Templates.HipChat.Presence.chatBar());o.find(".ajs-menu-bar").addClass("aui-buttons");o.find(".ajs-button > a").addClass("aui-button");o.find(".user-popup-more").addClass("aui-button");o.find(".user-popup-more > span > span").addClass("aui-icon aui-icon-small aui-iconfont-more").text("")}var q=d(".userLogoLink",m).attr("data-username");var j=c+encodeURIComponent(q);d.getJSON(j,e(k)).fail(function(t){if(t.status===404){try{var r=JSON.parse(t.responseText);if(r.userId){AJS.trigger("analyticsEvent",{name:"hipchat.presence.user.unknown",data:{userId:r.userId}})}}catch(s){}}})})}})})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-collaborative-editor-plugin:initialise-metrics-collection', location = '/js/initialise-metrics.js' */
define("collaborative-editing-initialise-metrics-collection",["confluence/legacy"],function(a){return function(){var b={"confluence.editor":!0,"confluence.editor.preload":!0,"confluence.editor.quick.fetchContent":!0,"confluence.editor.tinymce":!0,"confluence.editor.synchrony":!0,"confluence.editor.synchrony.CR":!0,"confluence.editor.synchrony.connect":!0,"confluence.editor.synchrony.deps":!0,"confluence.editor.synchrony.init":!0,"confluence.editor.synchrony.jsLoad":!0,"confluence.editor.synchrony.snapshot":!0,
"confluence.editor.synchrony.unmarshal":!0},c={"confluence.editor.synchrony.connect":!0};a.registerPerformanceSession&&a.registerPerformanceSession("confluence.editor.quickedit.loading.times",b,c)}});require("confluence/module-exporter").safeRequire("collaborative-editing-initialise-metrics-collection",function(a){a()});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-create-content-plugin:blueprint-first-time-tooltip-resources', location = 'com/atlassian/confluence/plugins/createcontent/js/first-time-tooptip.js' */
AJS.bind("sidebar.finished-loading",function(){var a=AJS.Meta.get("blueprint-index-popup-key");AJS.debug("Index key for "+a);a&&Confluence.Blueprint.showIndexPagePopup(a)});
Confluence.Blueprint=AJS.$.extend(Confluence.Blueprint,{showIndexPagePopup:function(a){var b=AJS.$(AJS.$("li.blueprint."+a)[0]);a=b.text();b=AJS.$(".icon",b);var c=AJS.InlineDialog(b.is(":visible")?b:AJS.$(".acs-nav-sections .quick-links-section"),"blueprintIndexSidebarPopup",function(a){return function(b,c,d){b.html(Confluence.Templates.Blueprints.sidebarIndexPagePopup({indexPageTitle:a.toLowerCase()}));d()}}(a),{addActiveClass:!1,hideDelay:null,noBind:!0});AJS.$(document).bind("showLayer",function(a){AJS.$("body").unbind("click.blueprintIndexSidebarPopup.inline-dialog-check")});
c.show();(function(a){AJS.$(document).on("click","#dismiss-index-popup",function(){a.hide();return!1})})(c);AJS.bind("quickedit.success",function(){c.hide()})}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-create-content-plugin:blueprint-first-time-tooltip-resources', location = 'com/atlassian/confluence/plugins/createcontent/soy/sidebar-index-page-popup.soy' */
// This file was automatically generated from sidebar-index-page-popup.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Blueprints.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Blueprints == 'undefined') { Confluence.Templates.Blueprints = {}; }


Confluence.Templates.Blueprints.sidebarIndexPagePopup = function(opt_data, opt_ignored) {
  return '<h2>' + soy.$$escapeHtml(AJS.format('Find your {0} here',opt_data.indexPageTitle)) + '</h2><p>' + soy.$$escapeHtml(AJS.format('You\x27\x27ve created a {0} page. A shortcut in your sidebar has been added where you can find other {0} in this space.',opt_data.indexPageTitle)) + '</p><br/><form>' + aui.buttons.button({text: 'Dismiss', id: 'dismiss-index-popup'}) + '</form>';
};
if (goog.DEBUG) {
  Confluence.Templates.Blueprints.sidebarIndexPagePopup.soyTemplateName = 'Confluence.Templates.Blueprints.sidebarIndexPagePopup';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-create-content-plugin:create-from-template-resources', location = 'com/atlassian/confluence/plugins/createcontent/js/create-from-template-macro.js' */
AJS.toInit(function(b){b(document).tooltip({live:'.create-from-template-button[aria-disabled\x3d"true"]',gravity:"n",title:"data-tooltip",delayIn:250,delayOut:0});b(".create-from-template-button").each(function(){var a=b(this);"true"!==a.attr("aria-disabled")&&a.click(function(){a.addClass("launching-dialog");Confluence.Blueprint.loadDialogAndOpenTemplate(a.data());return!1})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-create-content-plugin:create-from-template-resources', location = 'com/atlassian/confluence/plugins/createcontent/soy/create-from-template-macro.soy' */
// This file was automatically generated from create-from-template-macro.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Blueprints.CreateFromTemplate.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Blueprints == 'undefined') { Confluence.Templates.Blueprints = {}; }
if (typeof Confluence.Templates.Blueprints.CreateFromTemplate == 'undefined') { Confluence.Templates.Blueprints.CreateFromTemplate = {}; }


Confluence.Templates.Blueprints.CreateFromTemplate.macroTemplate = function(opt_data, opt_ignored) {
  return '<a class=\'aui-button create-from-template-button\'' + ((! opt_data.hasCreatePermission) ? 'aria-disabled=\'true\' data-tooltip="' + soy.$$escapeHtml('Sorry, you don\x27t have permission to create content. Contact your space administrator to request access.') + '"' : '') + 'data-space-key=\'' + soy.$$escapeHtml(opt_data.spaceKey) + '\' href=\'' + soy.$$escapeHtml(opt_data.createContentUrl) + '\'' + ((opt_data.title) ? 'data-title=\'' + soy.$$escapeHtml(opt_data.title) + '\'' : '') + ((opt_data.templateId) ? 'data-template-id=\'' + soy.$$escapeHtml(opt_data.templateId) + '\'' : '') + ((opt_data.contentBlueprintId) ? 'data-content-blueprint-id=\'' + soy.$$escapeHtml(opt_data.contentBlueprintId) + '\'' : '') + '>' + soy.$$escapeHtml(opt_data.buttonLabel) + '</a>';
};
if (goog.DEBUG) {
  Confluence.Templates.Blueprints.CreateFromTemplate.macroTemplate.soyTemplateName = 'Confluence.Templates.Blueprints.CreateFromTemplate.macroTemplate';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.contributors:contributors-web-resources', location = 'com/atlassian/confluence/contributors/scripts/contributors.js' */
require(["ajs","underscore"],function(c,f){c.toInit(function(b){b("div.contributors-macro-ajax-container").each(function(){var a=b(this),d=b.parseJSON(f.unescape(a.find(".contributors-macro-parameters")[0].innerHTML));a.text("Generating contributors information...");b.ajax({dataType:"json",url:Confluence.getContextPath()+"/rest/com.atlassian.confluence.contributors/1.0/contributors",data:d,success:function(e){e.errorMessage?a.text(e.errorMessage):(a.html(Confluence.ContributorsMacro.renderContent(e)),
a.find(".show-hidden-contributors").click(function(){a.find(".hidden-contributor").removeClass("hidden");b(this).parent().remove();return!1}))},error:function(b,f,d){a.text("Failed to retrieve contributors information"+": "+d)}})})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.contributors:contributors-web-resources', location = 'com/atlassian/confluence/contributors/templates/contributors-macro.soy' */
// This file was automatically generated from contributors-macro.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.ContributorsMacro.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.ContributorsMacro == 'undefined') { Confluence.ContributorsMacro = {}; }


Confluence.ContributorsMacro.ajaxContainer = function(opt_data, opt_ignored) {
  return '<div class="contributors-macro-ajax-container"><div style="display; none;" class="contributors-macro-parameters">' + soy.$$escapeHtml(opt_data.macroParameters) + '</div></div>';
};
if (goog.DEBUG) {
  Confluence.ContributorsMacro.ajaxContainer.soyTemplateName = 'Confluence.ContributorsMacro.ajaxContainer';
}


Confluence.ContributorsMacro.renderContent = function(opt_data, opt_ignored) {
  return '<div class="plugin-contributors">' + ((opt_data.layoutStyle == 'FLAT') ? Confluence.ContributorsMacro.flatLayout(opt_data) : (opt_data.layoutStyle == 'LIST') ? Confluence.ContributorsMacro.listLayout(opt_data) : '<span>Unknown layout style</span>') + '</div>';
};
if (goog.DEBUG) {
  Confluence.ContributorsMacro.renderContent.soyTemplateName = 'Confluence.ContributorsMacro.renderContent';
}


Confluence.ContributorsMacro.flatLayout = function(opt_data, opt_ignored) {
  return '<div class="plugin-contributors"><span><span>' + Confluence.ContributorsMacro.flatContributorsList({contributors: opt_data.visibleContributors, showCount: opt_data.showCount, showTime: opt_data.showTime}) + ((opt_data.hiddenContributors.length > 0) ? ',' : '') + '</span>' + ((opt_data.hiddenContributors.length > 0) ? '<span class="hidden hidden-contributor">' + Confluence.ContributorsMacro.flatContributorsList({contributors: opt_data.hiddenContributors, showCount: opt_data.showCount, showTime: opt_data.showTime}) + '</span><span><a href="#" class="show-hidden-contributors" title="' + soy.$$escapeHtml(AJS.format('{0} more...',opt_data.hiddenContributors.length)) + '">...</a></span>' : '') + '</span></div>';
};
if (goog.DEBUG) {
  Confluence.ContributorsMacro.flatLayout.soyTemplateName = 'Confluence.ContributorsMacro.flatLayout';
}


Confluence.ContributorsMacro.listLayout = function(opt_data, opt_ignored) {
  var output = '<div class="plugin-contributors"><ul class="contributors-list">';
  var contributorList39 = opt_data.visibleContributors;
  var contributorListLen39 = contributorList39.length;
  for (var contributorIndex39 = 0; contributorIndex39 < contributorListLen39; contributorIndex39++) {
    var contributorData39 = contributorList39[contributorIndex39];
    output += '<li>' + Confluence.ContributorsMacro.contributor({contributor: contributorData39, showCount: opt_data.showCount, showTime: opt_data.showTime}) + '</li>';
  }
  if (opt_data.hiddenContributors.length > 0) {
    output += '<li><a href="#" class="show-hidden-contributors" title="' + soy.$$escapeHtml(AJS.format('{0} more...',opt_data.hiddenContributors.length)) + '">...</a></li>';
    var contributorList52 = opt_data.hiddenContributors;
    var contributorListLen52 = contributorList52.length;
    for (var contributorIndex52 = 0; contributorIndex52 < contributorListLen52; contributorIndex52++) {
      var contributorData52 = contributorList52[contributorIndex52];
      output += '<li class="hidden hidden-contributor">' + Confluence.ContributorsMacro.contributor({contributor: contributorData52, showCount: opt_data.showCount, showTime: opt_data.showTime}) + '</li>';
    }
  }
  output += '</ul></div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.ContributorsMacro.listLayout.soyTemplateName = 'Confluence.ContributorsMacro.listLayout';
}


Confluence.ContributorsMacro.flatContributorsList = function(opt_data, opt_ignored) {
  var output = '';
  var contributorList62 = opt_data.contributors;
  var contributorListLen62 = contributorList62.length;
  for (var contributorIndex62 = 0; contributorIndex62 < contributorListLen62; contributorIndex62++) {
    var contributorData62 = contributorList62[contributorIndex62];
    output += ((! (contributorIndex62 == 0)) ? ',' : '') + Confluence.ContributorsMacro.contributor({contributor: contributorData62, showCount: opt_data.showCount, showTime: opt_data.showTime});
  }
  return output;
};
if (goog.DEBUG) {
  Confluence.ContributorsMacro.flatContributorsList.soyTemplateName = 'Confluence.ContributorsMacro.flatContributorsList';
}


Confluence.ContributorsMacro.contributor = function(opt_data, opt_ignored) {
  return Confluence.Templates.User.usernameLink({username: opt_data.contributor.idString, fullName: opt_data.contributor.fullNameString, canView: false}) + ' ' + ((opt_data.showCount) ? soy.$$escapeHtml(opt_data.contributor.totalCount) : '') + ' ' + ((opt_data.showTime) ? '(' + soy.$$escapeHtml(opt_data.contributor.relativeLastActiveTimeStr) + ')' : '');
};
if (goog.DEBUG) {
  Confluence.ContributorsMacro.contributor.soyTemplateName = 'Confluence.ContributorsMacro.contributor';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.issue-status-plugin:issue-status-resources', location = 'templates/status.soy' */
// This file was automatically generated from status.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace JIRA.Template.Util.Issue.Status.
 */

if (typeof JIRA == 'undefined') { var JIRA = {}; }
if (typeof JIRA.Template == 'undefined') { JIRA.Template = {}; }
if (typeof JIRA.Template.Util == 'undefined') { JIRA.Template.Util = {}; }
if (typeof JIRA.Template.Util.Issue == 'undefined') { JIRA.Template.Util.Issue = {}; }
if (typeof JIRA.Template.Util.Issue.Status == 'undefined') { JIRA.Template.Util.Issue.Status = {}; }


JIRA.Template.Util.Issue.Status.issueStatusResolver = function(opt_data, opt_ignored) {
  return '' + ((! opt_data.issueStatus) ? '<span class="aui-icon aui-icon-small aui-iconfont-help jira-issue-status-render-error" title="' + soy.$$escapeHtml('No issue status information was provided') + '"></span>' : (opt_data.issueStatus.statusCategory) ? JIRA.Template.Util.Issue.Status.issueStatus(opt_data) : JIRA.Template.Util.Issue.Status.iconStatus({name: opt_data.issueStatus.name, iconUrl: opt_data.issueStatus.iconUrl, description: opt_data.issueStatus.description, isCompact: opt_data.isCompact}));
};
if (goog.DEBUG) {
  JIRA.Template.Util.Issue.Status.issueStatusResolver.soyTemplateName = 'JIRA.Template.Util.Issue.Status.issueStatusResolver';
}


JIRA.Template.Util.Issue.Status.iconStatus = function(opt_data, opt_ignored) {
  return '<img src="' + soy.$$escapeHtml(opt_data.iconUrl) + '" width="16" height="16" alt="' + soy.$$escapeHtml(opt_data.name) + '" title="' + soy.$$escapeHtml(opt_data.name) + ((opt_data.description) ? ' - ' + soy.$$escapeHtml(opt_data.description) : '') + '" class="jira-issue-status-icon">' + ((! opt_data.isCompact) ? ' ' + soy.$$escapeHtml(opt_data.name) : '');
};
if (goog.DEBUG) {
  JIRA.Template.Util.Issue.Status.iconStatus.soyTemplateName = 'JIRA.Template.Util.Issue.Status.iconStatus';
}


JIRA.Template.Util.Issue.Status.issueStatus = function(opt_data, opt_ignored) {
  return '' + JIRA.Template.Util.Issue.Status.statusLozenge({name: opt_data.issueStatus.name, categoryKey: opt_data.issueStatus.statusCategory.key, colorName: opt_data.issueStatus.statusCategory.colorName, description: opt_data.issueStatus.description, isSubtle: opt_data.isSubtle, isCompact: opt_data.isCompact, maxWidth: opt_data.maxWidth});
};
if (goog.DEBUG) {
  JIRA.Template.Util.Issue.Status.issueStatus.soyTemplateName = 'JIRA.Template.Util.Issue.Status.issueStatus';
}


JIRA.Template.Util.Issue.Status.statusLozenge = function(opt_data, opt_ignored) {
  var output = '';
  var maxWidth__soy46 = opt_data.maxWidth ? opt_data.maxWidth : 'medium';
  var tooltipContent__soy47 = '<span class="jira-issue-status-tooltip-title">' + soy.$$escapeHtml(opt_data.name) + '</span>' + ((opt_data.description) ? '<br><span class="jira-issue-status-tooltip-desc">' + soy.$$escapeHtml(opt_data.description) + '</span>' : '');
  output += '<span class=" jira-issue-status-lozenge aui-lozenge ' + JIRA.Template.Util.Issue.Status.statusLozengeClasses(opt_data) + ((opt_data.isSubtle && ! opt_data.isCompact) ? ' aui-lozenge-subtle' : '') + ((opt_data.isCompact) ? ' jira-issue-status-lozenge-compact' : '') + ' jira-issue-status-lozenge-max-width-' + soy.$$escapeHtml(maxWidth__soy46) + '" data-tooltip="' + soy.$$escapeHtml(tooltipContent__soy47) + '">' + soy.$$escapeHtml(opt_data.name) + '</span>';
  return output;
};
if (goog.DEBUG) {
  JIRA.Template.Util.Issue.Status.statusLozenge.soyTemplateName = 'JIRA.Template.Util.Issue.Status.statusLozenge';
}


JIRA.Template.Util.Issue.Status.statusLozengeClasses = function(opt_data, opt_ignored) {
  return 'jira-issue-status-lozenge-' + soy.$$escapeHtml(opt_data.colorName ? opt_data.colorName : 'medium-gray') + ' ' + ((opt_data.categoryKey) ? 'jira-issue-status-lozenge-' + soy.$$escapeHtml(opt_data.categoryKey) : '');
};
if (goog.DEBUG) {
  JIRA.Template.Util.Issue.Status.statusLozengeClasses.soyTemplateName = 'JIRA.Template.Util.Issue.Status.statusLozengeClasses';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.issue-status-plugin:issue-status-resources', location = '/js/issue-status-plugin.js' */
AJS.$(function(){if(AJS.$.fn.tooltip){AJS.$(".jira-issue-status-lozenge[data-tooltip]").tooltip({aria:true,gravity:AJS.$.fn.tipsy.autoWE,delayIn:100,html:true,live:true,title:"data-tooltip",className:"jira-issue-status-tooltip"})}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-jira-metadata:confluence-jira-metadata-resources', location = 'soy/jira-metadata.soy' */
// This file was automatically generated from jira-metadata.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Metadata.Jira.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Metadata == 'undefined') { Confluence.Templates.Metadata = {}; }
if (typeof Confluence.Templates.Metadata.Jira == 'undefined') { Confluence.Templates.Metadata.Jira = {}; }


Confluence.Templates.Metadata.Jira.metadata = function(opt_data, opt_ignored) {
  var output = '<div id="jira-metadata-dialog" class="rendered-content"><h2 class="title">' + soy.$$escapeHtml('Jira links') + '</h2><div class="items-section">';
  var groupList6 = opt_data.groups;
  var groupListLen6 = groupList6.length;
  for (var groupIndex6 = 0; groupIndex6 < groupListLen6; groupIndex6++) {
    var groupData6 = groupList6[groupIndex6];
    if (groupData6.items.length) {
      switch (groupData6.type) {
        case 'ISSUES':
          output += Confluence.Templates.Metadata.Jira.renderGroup({items: groupData6.items, headingText: 'Issues', type: groupData6.type, links: groupData6.links});
          break;
        case 'SPRINTS':
          output += Confluence.Templates.Metadata.Jira.renderGroup({items: groupData6.items, headingText: 'Sprints', type: groupData6.type, links: groupData6.links});
          break;
        case 'VERSIONS':
          output += Confluence.Templates.Metadata.Jira.renderGroup({items: groupData6.items, headingText: 'Versions', type: groupData6.type, links: groupData6.links});
          break;
        case 'EPICS':
          output += Confluence.Templates.Metadata.Jira.renderGroup({items: groupData6.items, headingText: 'Epics', type: groupData6.type, links: groupData6.links});
          break;
      }
    }
  }
  output += '</div>' + Confluence.Templates.Metadata.Jira.renderAuthPrompts({appLinks: opt_data.unauthorisedAppLinks}) + Confluence.Templates.Metadata.Jira.renderJiraErrors(opt_data) + '</div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Metadata.Jira.metadata.soyTemplateName = 'Confluence.Templates.Metadata.Jira.metadata';
}


Confluence.Templates.Metadata.Jira.featureDiscovery = function(opt_data, opt_ignored) {
  return '<div id="jira-metadata-feature-discovery"><h2>' + soy.$$escapeHtml('View related Jira items here') + '</h2><p>' + soy.$$escapeHtml('Now you can see which epics, sprints, versions and issues relate to this page.') + '</p><div class="aui-toolbar2" role="toolbar"><div class="aui-toolbar2-inner">' + aui.buttons.button({text: 'Show me', extraClasses: 'showme'}) + aui.buttons.button({text: 'Don\x27t show again', type: 'link', extraClasses: 'close'}) + '</div></div></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Metadata.Jira.featureDiscovery.soyTemplateName = 'Confluence.Templates.Metadata.Jira.featureDiscovery';
}


Confluence.Templates.Metadata.Jira.nometadata = function(opt_data, opt_ignored) {
  return '<div id="jira-metadata-dialog" class="rendered-content">' + aui.message.warning({content: '<p>' + soy.$$escapeHtml('Jira links cannot be displayed. Either you do not have correct Jira permissions or the links have been removed.') + '</p>'}) + '</div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Metadata.Jira.nometadata.soyTemplateName = 'Confluence.Templates.Metadata.Jira.nometadata';
}


Confluence.Templates.Metadata.Jira.renderAuthPrompts = function(opt_data, opt_ignored) {
  var output = '';
  if (opt_data.appLinks.length) {
    var param66 = '';
    if (opt_data.appLinks.length == 1) {
      var appLink__soy69 = opt_data.appLinks[0];
      param66 += '<p>' + soy.$$filterNoAutoescape(AJS.format('{0}Login \x26amp; Approve{1} to retrieve data from {2}','<a class="jira-metadata-auth-link" href="#" data-href="' + appLink__soy69.authorisationUrl + '">','</a>',appLink__soy69.htmlSafeName)) + '</p>';
    } else {
      param66 += '<p>' + soy.$$escapeHtml('Authenticate to retrieve data from the following instances:') + '</p>';
      var appLinkList78 = opt_data.appLinks;
      var appLinkListLen78 = appLinkList78.length;
      for (var appLinkIndex78 = 0; appLinkIndex78 < appLinkListLen78; appLinkIndex78++) {
        var appLinkData78 = appLinkList78[appLinkIndex78];
        param66 += '<div><a class="jira-metadata-auth-link" href="#" data-href="' + soy.$$escapeHtml(appLinkData78.authorisationUrl) + '">' + soy.$$escapeHtml(appLinkData78.name) + '</a></div>';
      }
    }
    output += aui.message.hint({content: param66});
  }
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Metadata.Jira.renderAuthPrompts.soyTemplateName = 'Confluence.Templates.Metadata.Jira.renderAuthPrompts';
}


Confluence.Templates.Metadata.Jira.renderGroup = function(opt_data, opt_ignored) {
  var output = '<div class="jira-metadata-section ' + soy.$$escapeHtml(opt_data.type) + '-section"><div class="section-label"><span class="icon"></span><span>' + soy.$$escapeHtml(opt_data.headingText) + '</span></div><ul class="jira-metadata-list jira-' + soy.$$escapeHtml(opt_data.type) + '-list">';
  var itemList94 = opt_data.items;
  var itemListLen94 = itemList94.length;
  for (var itemIndex94 = 0; itemIndex94 < itemListLen94; itemIndex94++) {
    var itemData94 = itemList94[itemIndex94];
    output += '<li class="jira-metadata-item"><span class="item-label"><a href="' + soy.$$escapeHtml("/wiki") + '/plugins/servlet/jira-metadata/redirect?u=' + soy.$$escapeUri(itemData94.url) + '&t=' + soy.$$escapeHtml(opt_data.type) + '" title="' + soy.$$escapeHtml(itemData94.name) + '">' + soy.$$escapeHtml(itemData94.name) + '</a>' + ((itemData94.status) ? '&nbsp;' + ((itemData94.status.statusCategory) ? JIRA.Template.Util.Issue.Status.issueStatusResolver({issueStatus: itemData94.status, isSubtle: true}) : '<span class="item-status">(' + soy.$$escapeHtml(itemData94.status.name) + ')</span>') : '') + '</span>' + ((itemData94.description != '') ? '<span class="item-subtext">' + soy.$$escapeHtml(itemData94.description) + '</span>' : '') + '</li>';
  }
  output += '</ul><ul class="jira-metadata-list ' + soy.$$escapeHtml(opt_data.type) + '-more-link">';
  var linkList130 = opt_data.links;
  var linkListLen130 = linkList130.length;
  for (var linkIndex130 = 0; linkIndex130 < linkListLen130; linkIndex130++) {
    var linkData130 = linkList130[linkIndex130];
    output += '<li class="jira-metadata-item"><a href="' + soy.$$escapeHtml("/wiki") + '/plugins/servlet/jira-metadata/redirect?u=' + soy.$$escapeUri(linkData130.url) + '&t=' + soy.$$escapeHtml(opt_data.type) + '&more">' + soy.$$escapeHtml(AJS.format('View {0} more in {1}',linkData130.numItems,linkData130.appName)) + '</a></li>';
  }
  output += '</ul></div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Metadata.Jira.renderGroup.soyTemplateName = 'Confluence.Templates.Metadata.Jira.renderGroup';
}


Confluence.Templates.Metadata.Jira.loadingMetadata = function(opt_data, opt_ignored) {
  return '<div id="jira-metadata-dialog"><h2 class="title">' + soy.$$escapeHtml('Jira links') + '</h2><div class="spinner-container"><div class="spinner"></div></div></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Metadata.Jira.loadingMetadata.soyTemplateName = 'Confluence.Templates.Metadata.Jira.loadingMetadata';
}


Confluence.Templates.Metadata.Jira.renderJiraErrors = function(opt_data, opt_ignored) {
  var output = '';
  if (opt_data.errors.length == 1) {
    var error__soy150 = opt_data.errors[0];
    output += aui.message.warning({content: '<p>' + soy.$$escapeHtml('Unable to retrieve Jira metadata.') + ' ' + soy.$$escapeHtml(error__soy150.errorMessage) + '</p>'});
  } else if (opt_data.errors.length > 1) {
    var param159 = '<p>' + soy.$$escapeHtml('Unable to retrieve Jira metadata. The following errors occurred:') + '</p><ul>';
    var errorList163 = opt_data.errors;
    var errorListLen163 = errorList163.length;
    for (var errorIndex163 = 0; errorIndex163 < errorListLen163; errorIndex163++) {
      var errorData163 = errorList163[errorIndex163];
      param159 += '<li>' + soy.$$escapeHtml(errorData163.errorMessage) + '</li>';
    }
    param159 += '</ul>';
    output += aui.message.warning({content: param159});
  }
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Metadata.Jira.renderJiraErrors.soyTemplateName = 'Confluence.Templates.Metadata.Jira.renderJiraErrors';
}


Confluence.Templates.Metadata.Jira.unknownError = function(opt_data, opt_ignored) {
  return '<div id="jira-metadata-dialog" class="rendered-content">' + aui.message.warning({content: '<p>' + soy.$$escapeHtml('Unable to retrieve Jira metadata. Could not connect to Confluence') + '</p>'}) + '</div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Metadata.Jira.unknownError.soyTemplateName = 'Confluence.Templates.Metadata.Jira.unknownError';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-jira-metadata:confluence-jira-metadata-resources', location = '/js/jira-metadata.js' */
AJS.toInit(function(c){function l(a,e){m(a,e);d.removeClass("hidden");d.attr("href")||(h=AJS.InlineDialog(d,"jira-metadata-dialog",function(f,u,n){AJS.trigger("analytics",{name:"confluence.jira.metadata.expanded"});b&&p?n():(b=f,n(),q(f));return!1},{hideDelay:null}),d.click(function(){c("#jira-metadata-dialog").is(":visible")&&h.hide()}));d&&!AJS.Meta.get("blueprint-index-popup-key")&&Confluence.FeatureDiscovery.forPlugin("com.atlassian.confluence.plugins.confluence-jira-metadata").shouldShow("linked-issues-dropdown")&&
(r(),d.one("click",function(){Confluence.FeatureDiscovery.forPlugin("com.atlassian.confluence.plugins.confluence-jira-metadata").markDiscovered("linked-issues-dropdown")}))}function m(a,e){e||(e=c("\x3cspan/\x3e").addClass("aui-icon aui-icon-small aui-iconfont-jira"),c("#content-metadata-jira").text(1==a?"1 Jira link":AJS.format("{0} Jira links",a)).prepend(e))}function r(){g=AJS.InlineDialog(d,"jira-metadata-discovery",function(a,e,f){a.html(Confluence.Templates.Metadata.Jira.featureDiscovery());
a.find(".showme").on("click",function(){Confluence.FeatureDiscovery.forPlugin("com.atlassian.confluence.plugins.confluence-jira-metadata").markDiscovered("linked-issues-dropdown");g.hide();h.show()});a.find(".close").on("click",function(){Confluence.FeatureDiscovery.forPlugin("com.atlassian.confluence.plugins.confluence-jira-metadata").markDiscovered("linked-issues-dropdown");g.hide()});f()},{noBind:!0,closeOthers:!1,hideDelay:null});g.show();Confluence.FeatureDiscovery.forPlugin("com.atlassian.confluence.plugins.confluence-jira-metadata").addDiscoveryView("linked-issues-dropdown")}
function q(){b&&0<b.height()&&b.css("height",b.height());b.html(Confluence.Templates.Metadata.Jira.loadingMetadata());b.find(".spinner").spin("medium");c.ajax({url:AJS.contextPath()+"/rest/jira-metadata/1.0/metadata?pageId\x3d"+AJS.Meta.get("page-id"),type:"GET",dataType:"json",contentType:"application/json",error:function(a){b&&b.css("height","");b.html(Confluence.Templates.Metadata.Jira.unknownError())},success:function(a){b&&b.css("height","");p=!0;m(a.count,!1);0!==a.count||a.unauthorisedAppLinks&&
0<a.unauthorisedAppLinks.length||0!=a.errors.length?a=Confluence.Templates.Metadata.Jira.metadata(a):(AJS.trigger("analytics",{name:"confluence.jira.metadata.error.no-metadata"}),a=Confluence.Templates.Metadata.Jira.nometadata());b.html(a);setTimeout(function(){b.find("#jira-metadata-dialog").addClass("show")},0);t()},complete:function(){c("#jira-metadata-dialog .icon-close").click(function(a){a.stopPropagation();c(this).closest(".closable").remove()})}})}function t(){c(".jira-metadata-auth-link").click(function(a){a.preventDefault();
AppLinks.authenticateRemoteCredentials(c(this).data("href"),q,function(){})})}var p=!1,h,b,d=c("#content-metadata-jira"),g,k=AJS.Meta.get("jira-metadata-count");0<k?l(k,AJS.Meta.get("jira-metadata-count-incomplete")):-1==k&&c.ajax({url:AJS.contextPath()+"/rest/jira-metadata/1.0/metadata/aggregate?pageId\x3d"+AJS.Meta.get("page-id"),type:"GET",dataType:"json",contentType:"application/json",cache:!1,success:function(a){0<a.count&&l(a.count,a.incomplete)}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-jira-content:confluence-jira-content-loader', location = 'com/atlassian/confluence/plugins/createjiracontent/js/page-helper.js' */
AJS.toInit(function(e){var h="Loading...",f=!1,c=e("div.jira-issues-created");if(0<c.length){if(window.history&&window.history.replaceState){var d=window.location.href;d=d.substr(0,d.indexOf("JIRAIssuesCreated")-1);window.history.replaceState({},document.title,d)}var k=c.find("#jira-content-message-panel-error-warning"),g=c.find("#jira-content-message-panel-view-more-link");g.click(function(a){a.preventDefault();g.hide();k.show()});c.hasClass("success")&&
setTimeout(function(){c.hide()},1E4)}Confluence&&Confluence.HighlightAction&&Confluence.HighlightAction.registerButtonHandler("com.atlassian.confluence.plugins.confluence-jira-content:create-Jira-issue-summary",{onClick:function(a){if(!f){var b=e("\x3cdiv\x3e");Confluence.CreateJiraContent.Dialogs.appendDialogTarget(a.area.average,b);dialog=Confluence.ScrollingInlineDialog(b,"create-issue-loading-dialog",function(l,n,m){l.html('\x3cspan class\x3d"aui-icon aui-icon-wait"\x3e\x3c/span\x3e '+h);m();
return!1},{});dialog.show();f=!0}WRM.require("wr!com.atlassian.confluence.plugins.confluence-jira-content:confluence-jira-content-resources").done(function(){Confluence.CreateJiraContent.FeatureDiscovery.shouldShowFeatureDiscovery()?Confluence.CreateJiraContent.Dialogs.showFeatureDiscoveryDialog(a):Confluence.CreateJiraContent.Dialogs.showCreateIssueDialog(a);"undefined"!==typeof b&&b.remove()})},shouldDisplay:Confluence.HighlightAction.WORKING_AREA.MAINCONTENT_ONLY});Confluence.CreateJiraContent=
{Dialogs:{}};Confluence.CreateJiraContent.Dialogs.appendDialogTarget=function(a,b){Confluence.DocThemeUtils.appendAbsolutePositionedElement(b);b.css({top:a.top,height:a.height,left:a.left,width:a.width,"z-index":-9999,position:"absolute"}).addClass("confluence-jira-content-dialog-target")}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:soy-resources', location = 'soy/sidebar.soy' */
// This file was automatically generated from sidebar.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Sidebar.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Sidebar == 'undefined') { Confluence.Templates.Sidebar = {}; }


Confluence.Templates.Sidebar.headerStyles = function(opt_data, opt_ignored) {
  opt_data = opt_data || {};
  return '' + ((opt_data.sidebarWidth) ? '<style>.ia-fixed-sidebar, .ia-splitter-left {width: ' + soy.$$escapeHtml(opt_data.sidebarWidth) + 'px;}.theme-default .ia-splitter #main {margin-left: ' + soy.$$escapeHtml(opt_data.sidebarWidth) + 'px;}.ia-fixed-sidebar {visibility: hidden;}</style>' : '');
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.headerStyles.soyTemplateName = 'Confluence.Templates.Sidebar.headerStyles';
}


Confluence.Templates.Sidebar.sidebar = function(opt_data, opt_ignored) {
  return '<div class="acs-side-bar ia-scrollable-section"><div class="acs-side-bar-space-info tipsy-enabled" data-configure-tooltip="' + soy.$$escapeHtml('Edit space details') + '"><div class="avatar"><div class="space-logo" data-key="' + soy.$$escapeHtml(opt_data.space.key) + '" data-name="' + soy.$$escapeHtml(opt_data.space.name) + '" data-entity-type="confluence.space"><div class="avatar-img-container"><div class="avatar-img-wrapper"><a href="' + soy.$$escapeHtml(opt_data.space.homeUrl) + '" title="' + soy.$$escapeHtml(opt_data.space.name) + '"><img class="avatar-img" src="' + soy.$$escapeHtml(opt_data.space.logoUrl) + '" alt="' + soy.$$escapeHtml(opt_data.space.name) + '"></a></div></div></div></div><div class="space-information-container"><div class="name"><a href="' + soy.$$escapeHtml(opt_data.space.homeUrl) + '" title="' + soy.$$escapeHtml(opt_data.space.name) + '">' + soy.$$escapeHtml(opt_data.space.name) + '</a></div><div class="flyout-handle icon aui-icon aui-icon-small aui-iconfont-edit"></div>' + ((opt_data.hasFavouriteSpacePermission || opt_data.accessMode == 'READ_ONLY') ? '<div class="favourite-space-icon ' + ((opt_data.accessMode == 'READ_ONLY') ? 'disabled' : '') + '">' + Confluence.Templates.Sidebar.renderFavouriteSpace(opt_data) + '</div>' : '') + '</div></div><div class="acs-side-bar-content"><div class="acs-nav-wrapper"><div class="acs-nav" data-has-create-permission="' + soy.$$escapeHtml(opt_data.hasCreatePermission) + '" data-quick-links-state="' + soy.$$escapeHtml(opt_data.quickLinksState) + '" data-page-tree-state="' + soy.$$escapeHtml(opt_data.pageTreeState) + '" data-nav-type="' + soy.$$escapeHtml(opt_data.navType) + '">' + Confluence.Templates.Sidebar.renderLinks(opt_data) + '</div></div>' + ((opt_data.contextualNav) ? Confluence.Templates.Sidebar.contextualNav(opt_data) : '') + '</div><div class="hidden"><a href="' + soy.$$escapeHtml(opt_data.space.browseSpaceUrl) + '" id="space-pages-link"></a><script type="text/x-template" title="logo-config-content"><h2>' + soy.$$escapeHtml('Space Details') + '</h2><div class="personal-space-logo-hint">' + soy.$$filterNoAutoescape(AJS.format('Your profile picture is used as the logo for your personal space. \x3ca href\x3d\x22{0}\x22 target\x3d\x22_blank\x22\x3eChange your profile picture\x3c/a\x3e.',"/wiki" + '/users/profile/editmyprofilepicture.action')) + '</div><\/script></div></div>' + Confluence.Templates.Sidebar.renderSpaceToolsSection({advancedLinks: opt_data.advancedLinks, hasConfigurePermission: opt_data.hasConfigurePermission, currentlyViewed: opt_data.collectorToHighlight == 'spacebar-advanced', pageTreeState: opt_data.pageTreeState});
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.sidebar.soyTemplateName = 'Confluence.Templates.Sidebar.sidebar';
}


Confluence.Templates.Sidebar.renderFavouriteSpace = function(opt_data, opt_ignored) {
  return '<button class="' + ((opt_data.isFavouriteSpace) ? 'space-favourite-hidden ' : '') + 'space-favourite aui-icon aui-icon-small aui-iconfont-unstar" id="space-favourite-add" title="' + soy.$$escapeHtml('Add to my spaces') + '" aria-pressed="false"></button><button class="' + ((! opt_data.isFavouriteSpace) ? 'space-favourite-hidden ' : '') + 'space-favourite aui-icon aui-icon-small aui-iconfont-star" id="space-favourite-remove" title="' + soy.$$escapeHtml('Remove from my spaces') + '" aria-pressed="false"></button>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.renderFavouriteSpace.soyTemplateName = 'Confluence.Templates.Sidebar.renderFavouriteSpace';
}


Confluence.Templates.Sidebar.renderLinks = function(opt_data, opt_ignored) {
  return '<div class="acs-nav-sections">' + ((opt_data.mainLinks.length) ? Confluence.Templates.Sidebar.renderLinksSection({links: opt_data.mainLinks, sectionClass: 'main-links-section', collectorToHighlight: opt_data.collectorToHighlight}) : '') + ((opt_data.quickLinksState != 'hide') ? '<div class="quick-links-wrapper">' + ((opt_data.quickLinks.length) ? '<h5 class="ia-quick-links-header-title">' + soy.$$escapeHtml('Space shortcuts') + '</h5>' + Confluence.Templates.Sidebar.renderLinksSection({links: opt_data.quickLinks, sectionClass: 'quick-links-section tipsy-enabled', collectorToHighlight: null}) : (opt_data.hasConfigurePermission) ? '<h5 class="ia-quick-links-header-title">' + soy.$$escapeHtml('Space shortcuts') + '</h5><p class="tip">' + soy.$$filterNoAutoescape(AJS.format('Here you can add shortcut links to the most important content for your team or project. \x3ca href\x3d\x22{0}\x22 class\x3d\x22{1}\x22\x3eConfigure sidebar\x3c/a\x3e.','','configure-sidebar')) + '</p>' : '') + '</div>' : '') + ((opt_data.hasSidebarCustomisation) ? Confluence.Templates.Sidebar.renderCustomContent(opt_data) : '') + '</div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.renderLinks.soyTemplateName = 'Confluence.Templates.Sidebar.renderLinks';
}


Confluence.Templates.Sidebar.renderCustomContent = function(opt_data, opt_ignored) {
  return '<div class="custom-sidebar"><div class="custom-sidebar-content"><div class="content">' + soy.$$filterNoAutoescape(opt_data.sidebarCustomisation) + '</div></div></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.renderCustomContent.soyTemplateName = 'Confluence.Templates.Sidebar.renderCustomContent';
}


Confluence.Templates.Sidebar.renderLinksSection = function(opt_data, opt_ignored) {
  var output = '';
  if (opt_data.links.length) {
    output += '<div class="' + soy.$$escapeHtml(opt_data.sectionClass) + ' ' + ((opt_data.highlightSection) ? ' current-section' : '') + '"><ul class="acs-nav-list">';
    var linkList138 = opt_data.links;
    var linkListLen138 = linkList138.length;
    for (var linkIndex138 = 0; linkIndex138 < linkListLen138; linkIndex138++) {
      var linkData138 = linkList138[linkIndex138];
      output += '<li class="acs-nav-item ' + soy.$$escapeHtml(linkData138.styleClass) + ((opt_data.collectorToHighlight && linkData138.collectorKey == opt_data.collectorToHighlight) ? ' current-item" aria-current="true' : '') + '"' + ((linkData138.collectorKey) ? ' data-collector-key="' + soy.$$escapeHtml(linkData138.collectorKey) + '"' : '') + '><a class="acs-nav-item-link tipsy-enabled" href="' + soy.$$escapeHtml(linkData138.url) + '" data-collapsed-tooltip="' + soy.$$escapeHtml(linkData138.tooltip) + '">' + ((linkData138.styleClass == 'pinned_attachment') ? '<span class="aui-icon aui-icon-small aui-iconfont-attachment"></span>' : '<span class="icon"></span>') + '<span class="acs-nav-item-label">' + soy.$$escapeHtml(linkData138.title) + '</span></a></li>';
    }
    output += '</ul></div>';
  }
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.renderLinksSection.soyTemplateName = 'Confluence.Templates.Sidebar.renderLinksSection';
}


Confluence.Templates.Sidebar.contextualNav = function(opt_data, opt_ignored) {
  return '<div class="ia-secondary-container tipsy-enabled" data-tree-type="' + ((opt_data.forBlogs) ? 'blogs' : (opt_data.forSettings) ? 'settings' : soy.$$escapeHtml(opt_data.navType)) + '">' + ((opt_data.pageTreeState != 'hide') ? (opt_data.forBlogs) ? '<div class="ia-secondary-header"><h5 class="ia-secondary-header-title blog"><span class="icon"></span><span class="label">' + soy.$$escapeHtml('Blog') + '</span></h5></div><div class="ia-secondary-content">' + Confluence.Templates.Sidebar.pagetreeList({pagetree: opt_data.contextualNav}) + '</div>' : (opt_data.forSettings) ? '<div class="ia-secondary-header"><h5 class="ia-secondary-header-title settings"><span class="label">' + soy.$$escapeHtml('Advanced') + '</span></h5></div><div class="ia-secondary-content">' + Confluence.Templates.Sidebar.pagetreeList({pagetree: opt_data.contextualNav}) + '</div>' : (opt_data.navType == 'page-tree') ? '<div class="ia-secondary-header"><h5 class="ia-secondary-header-title page-tree"><span class="icon"></span><span class="label">' + soy.$$escapeHtml('Page tree') + '</span></h5></div>' + ((opt_data.pageTreeEmpty && opt_data.hasCreatePermission) ? '<p class="tip">' + soy.$$filterNoAutoescape(AJS.format('Get started by adding some pages to this space. \x3ca href\x3d\x22{0}\x22 class\x3d\x22{1}\x22\x3eCreate page\x3c/a\x3e.',"/wiki" + '/pages/createpage.action?spaceKey=' + opt_data.space.key,'page-tree-create-child-page-link')) + '</p>' : '<div class="ia-secondary-content">' + soy.$$filterNoAutoescape(opt_data.contextualNav) + '</div>') : Confluence.Templates.Sidebar.Pages.renderPageContextualNav({pageContextualNav: opt_data.contextualNav, hasCreatePermission: opt_data.hasCreatePermission}) : '') + '</div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.contextualNav.soyTemplateName = 'Confluence.Templates.Sidebar.contextualNav';
}


Confluence.Templates.Sidebar.pagetreeList = function(opt_data, opt_ignored) {
  var output = '<ul class="' + ((opt_data.isSubtree) ? 'ia-subpagetree' : 'ia-pagetree') + '">';
  var itemList220 = opt_data.pagetree;
  var itemListLen220 = itemList220.length;
  for (var itemIndex220 = 0; itemIndex220 < itemListLen220; itemIndex220++) {
    var itemData220 = itemList220[itemIndex220];
    output += Confluence.Templates.Sidebar.pagetreeItem(itemData220);
  }
  output += '</ul>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.pagetreeList.soyTemplateName = 'Confluence.Templates.Sidebar.pagetreeList';
}


Confluence.Templates.Sidebar.throbber = function(opt_data, opt_ignored) {
  return '<div class="content-container"><div class="throbber-container"><div class="throbber"><div class="spinner"></div><span>' + soy.$$escapeHtml('Loading...') + '</span></div></div></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.throbber.soyTemplateName = 'Confluence.Templates.Sidebar.throbber';
}


Confluence.Templates.Sidebar.treeThrobber = function(opt_data, opt_ignored) {
  return '<ul class="ia-subpagetree"><li class="acs-tree-item leaf"><span class="node-title">' + soy.$$escapeHtml('Loading...') + '</span></li></ul>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.treeThrobber.soyTemplateName = 'Confluence.Templates.Sidebar.treeThrobber';
}


Confluence.Templates.Sidebar.pagetreeItem = function(opt_data, opt_ignored) {
  return '<li class="acs-tree-item' + ((opt_data.hasChildren) ? (opt_data.children.length) ? ' opened' : ' closed' : ' leaf') + ((opt_data.groupType) ? ' grouping' : '') + ((opt_data.active) ? ' current-item' : '') + '"' + ((opt_data.pageId) ? ' data-page-id="' + soy.$$escapeHtml(opt_data.pageId) + '"' : '') + ((opt_data.groupType) ? ' data-group-type="' + soy.$$escapeHtml(opt_data.groupType) + '" data-group-value="' + soy.$$escapeHtml(opt_data.groupValue) + '"' : '') + ((opt_data.hasChildren) ? ' aria-expanded="' + ((opt_data.children.length) ? 'true' : 'false') + '"' : '') + '>' + ((! opt_data.groupType) ? '<a href="' + soy.$$escapeHtml(opt_data.url) + '">' : '') + '<span class="icon aui-icon aui-icon-small ' + ((opt_data.hasChildren) ? (opt_data.children.length) ? 'aui-iconfont-expanded' : 'aui-iconfont-collapsed' : '') + '"></span><span class="node-title navigation-pseudo-link">' + soy.$$escapeHtml(opt_data.title) + '</span>' + ((! opt_data.groupType) ? '</a>' : '') + ((opt_data.children && opt_data.children.length > 0) ? Confluence.Templates.Sidebar.pagetreeList({pagetree: opt_data.children, isSubtree: true}) : '') + '</li>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.pagetreeItem.soyTemplateName = 'Confluence.Templates.Sidebar.pagetreeItem';
}


Confluence.Templates.Sidebar.renderSpaceToolsSection = function(opt_data, opt_ignored) {
  var output = '<div class="space-tools-section"><div id="space-tools-menu-additional-items" class="hidden">';
  var linkList299 = opt_data.advancedLinks;
  var linkListLen299 = linkList299.length;
  for (var linkIndex299 = 0; linkIndex299 < linkListLen299; linkIndex299++) {
    var linkData299 = linkList299[linkIndex299];
    output += '<div data-label="' + soy.$$escapeHtml(linkData299.title) + '" data-class="' + soy.$$escapeHtml(linkData299.styleClass) + '" data-href="' + soy.$$escapeHtml(linkData299.url) + '">' + soy.$$escapeHtml(linkData299.title) + '</div>';
  }
  output += ((opt_data.hasConfigurePermission) ? '<div data-label="' + soy.$$escapeHtml('Configure sidebar') + '" data-class="configure-sidebar" data-href="">' + soy.$$escapeHtml('Configure sidebar') + '</div>' : '') + '</div>' + aui.dropdown2.trigger({id: 'space-tools-menu-trigger', tagName: 'button', content: '<span class="aui-icon aui-icon-small aui-iconfont-configure">' + soy.$$escapeHtml('Configure') + '</span><span class="aui-button-label">' + soy.$$escapeHtml('Space tools') + '</span>', extraClasses: 'aui-button aui-button-subtle tipsy-enabled aui-dropdown2-trigger-arrowless ' + ((opt_data.currentlyViewed) ? ' current-item' : ''), menu: {id: 'space-tools-menu'}}) + aui.dropdown2.contents({id: 'space-tools-menu', extraClasses: 'aui-style-default space-tools-dropdown', extraAttributes: {'data-aui-alignment': 'top left'}}) + '<a href="#" role="button" class="expand-collapse-trigger aui-icon aui-icon-small ' + ((opt_data.pageTreeState == 'hide') ? 'aui-iconfont-chevron-double-right" aria-expanded="false"' : 'aui-iconfont-chevron-double-left" aria-expanded="true"') + '></a></div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.renderSpaceToolsSection.soyTemplateName = 'Confluence.Templates.Sidebar.renderSpaceToolsSection';
}


Confluence.Templates.Sidebar.spaceToolsMenu = function(opt_data, opt_ignored) {
  var output = '';
  if (opt_data.isAuiFiveSeven) {
    output += '<div class="space-tools-menu"><div class="aui-dropdown2-section"><ul class="space-tools-navigation">';
    var linkList348 = opt_data.spaceToolLinks;
    var linkListLen348 = linkList348.length;
    for (var linkIndex348 = 0; linkIndex348 < linkListLen348; linkIndex348++) {
      var linkData348 = linkList348[linkIndex348];
      output += '<li><a href="' + soy.$$escapeHtml(linkData348.href) + '" title="' + soy.$$escapeHtml(linkData348.label) + '">' + soy.$$escapeHtml(linkData348.label) + '</a></li>';
    }
    output += '</ul></div>';
    if (opt_data.spaceLinks.length > 0) {
      output += '<div class="aui-dropdown2-section"><ul class="space-operations">';
      var linkList361 = opt_data.spaceLinks;
      var linkListLen361 = linkList361.length;
      for (var linkIndex361 = 0; linkIndex361 < linkListLen361; linkIndex361++) {
        var linkData361 = linkList361[linkIndex361];
        output += '<li><a class="' + soy.$$escapeHtml(linkData361.className) + '" href="' + soy.$$escapeHtml(linkData361.href) + '" title="' + soy.$$escapeHtml(linkData361.label) + '">' + soy.$$escapeHtml(linkData361.label) + '</a></li>';
      }
      output += '</ul></div>';
    }
    output += '</div>';
  } else {
    output += aui.dropdown2.itemGroup({isTruncated: true, items: opt_data.spaceToolLinks, extraClasses: 'space-tools-navigation'}) + ((opt_data.spaceLinks.length > 0) ? aui.dropdown2.itemGroup({isTruncated: true, items: opt_data.spaceLinks, extraClasses: 'space-operations'}) : '');
  }
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.spaceToolsMenu.soyTemplateName = 'Confluence.Templates.Sidebar.spaceToolsMenu';
}


Confluence.Templates.Sidebar.configure = function(opt_data, opt_ignored) {
  return '<div class="acs-nav-sections"><table id="acs-nav-list-main" class="acs-nav-list"></table><div class="acs-nav-list-quick-section' + ((opt_data.quickLinksState == 'hide') ? ' hidden-section' : '') + '"><div class="quick-links-header"><h5>' + soy.$$escapeHtml('Space shortcuts') + '</h5><a href="#" class="aui-icon aui-icon-small toggle-link ' + ((opt_data.quickLinksState == 'hide') ? 'aui-iconfont-add-circle" aria-disabled="true"' : 'aui-iconfont-plan-disabled" aria-disabled="false"') + ' data-tooltip="' + soy.$$escapeHtml('Hide/Show space shortcuts') + '"></a></div><table id="acs-nav-list-quick" class="acs-nav-list"></table><p class="tip">' + soy.$$escapeHtml('Click \x22Add link\x22 to add links to the sidebar.') + '</p><a class="acs-add-link" href="#"><span class="aui-icon aui-icon-small aui-iconfont-add-small"></span><span class="label">' + soy.$$escapeHtml('Add link') + '</span></a></div>' + ((opt_data.hasSidebarCustomisation) ? '<p class="tip">' + soy.$$filterNoAutoescape(AJS.format('You can continue editing your customized sidebar in the space admin. \x3ca href\x3d\x22{0}\x22 class\x3d\x22{1}\x22\x3eGo to space admin\x3c/a\x3e.',opt_data.customContentAdminLink,'custom-sidebar-tip')) + '</p>' + Confluence.Templates.Sidebar.renderCustomContent(opt_data) : '') + '<div class="acs-nav-list-page-tree-section' + ((opt_data.pageTreeState == 'hide') ? ' hidden-section' : '') + '"><div class="page-tree-header"><h5>' + soy.$$escapeHtml('Navigation display options') + '</h5><a href="#" class="aui-icon aui-icon-small ' + ((opt_data.pageTreeState == 'hide') ? 'aui-iconfont-add-circle" aria-disabled="true"' : 'aui-iconfont-plan-disabled" aria-disabled="false"') + ' data-tooltip="' + soy.$$escapeHtml('Hide/Show navigation display options') + '"></a></div><form class="aui page-tree-options"><div class="radio"><input ' + ((opt_data.pageTreeState == 'hide') ? 'disabled ' : '') + ' class="radio acs-nav-type" type="radio" name="nav-type" value="pages" id="nav-type-pages" ' + ((! opt_data.pageTree) ? 'checked' : '') + '><label for="nav-type-pages">' + soy.$$escapeHtml('Child pages') + '</label></div><div class="radio"><input ' + ((opt_data.pageTreeState == 'hide') ? 'disabled ' : '') + ' class="radio acs-nav-type" type="radio" name="nav-type" value="page-tree" id="nav-type-tree" ' + ((opt_data.pageTree) ? 'checked' : '') + '><label for="nav-type-tree">' + soy.$$escapeHtml('Page tree') + '</label></div></form></div><button class="aui-style aui-button acs-done-link">' + soy.$$escapeHtml('Done') + '</button></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.configure.soyTemplateName = 'Confluence.Templates.Sidebar.configure';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:soy-resources', location = 'soy/sidebar-pages.soy' */
// This file was automatically generated from sidebar-pages.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Sidebar.Pages.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Sidebar == 'undefined') { Confluence.Templates.Sidebar = {}; }
if (typeof Confluence.Templates.Sidebar.Pages == 'undefined') { Confluence.Templates.Sidebar.Pages = {}; }


Confluence.Templates.Sidebar.Pages.renderPageContextualNav = function(opt_data, opt_ignored) {
  return '<div class="ia-secondary-header"><h5 class="ia-secondary-header-title pages"><span class="label">' + soy.$$escapeHtml('Child pages') + '</span></h5></div><div class="ia-secondary-parent-content">' + Confluence.Templates.Sidebar.Pages.parentPage({parentPage: opt_data.pageContextualNav.parentPage}) + '</div><div class="ia-secondary-current-content">' + Confluence.Templates.Sidebar.Pages.currentPage({currentPage: opt_data.pageContextualNav.currentPage}) + '</div><div class="ia-secondary-content">' + Confluence.Templates.Sidebar.Pages.childPages({children: opt_data.pageContextualNav, createPermission: opt_data.hasCreatePermission}) + '</div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.Pages.renderPageContextualNav.soyTemplateName = 'Confluence.Templates.Sidebar.Pages.renderPageContextualNav';
}


Confluence.Templates.Sidebar.Pages.childPages = function(opt_data, opt_ignored) {
  return '<div class="contextual-nav-child-pages">' + ((opt_data.children.initialChildPages.length) ? '<ul class="children">' + Confluence.Templates.Sidebar.Pages.renderChildren({children: opt_data.children.initialChildPages}) + '</ul>' + ((opt_data.children.remainingChildPages.length) ? '<ul class="more-children">' + Confluence.Templates.Sidebar.Pages.renderChildren({children: opt_data.children.remainingChildPages}) + '</ul><a class="more-children-link" href=""><span class="icon"></span><span class="label">' + soy.$$escapeHtml(AJS.format('{0} more child pages',opt_data.children.remainingChildPages.length)) + '</span></a>' : '') : '') + ((opt_data.createPermission && opt_data.children.createLink) ? '<a class="create-child-page-link" href="' + soy.$$escapeHtml(opt_data.children.createLink) + '"><span class="aui-icon aui-icon-small aui-iconfont-add-small"></span><span class="label">' + soy.$$escapeHtml('Create child page') + '</span></a>' : '') + '</div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.Pages.childPages.soyTemplateName = 'Confluence.Templates.Sidebar.Pages.childPages';
}


Confluence.Templates.Sidebar.Pages.currentPage = function(opt_data, opt_ignored) {
  return '' + ((opt_data.currentPage) ? '<ul class="ia-secondary-currentPage-title wiki' + ((opt_data.currentPage.active) ? ' current-item' : '') + '"><li><span class="icon"></span><span class="label">' + soy.$$escapeHtml(opt_data.currentPage.title) + '</span></li></ul>' : '');
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.Pages.currentPage.soyTemplateName = 'Confluence.Templates.Sidebar.Pages.currentPage';
}


Confluence.Templates.Sidebar.Pages.parentPage = function(opt_data, opt_ignored) {
  return '' + ((opt_data.parentPage) ? '<ul class="parent ia-secondary-header-title wiki' + ((opt_data.parentPage.active) ? ' current-item' : '') + '"><li class="parent-item"><a class="parent-item-link" href="' + soy.$$escapeHtml(opt_data.parentPage.url) + '" title="' + soy.$$escapeHtml(opt_data.parentPage.title) + '"><span class="icon"></span><span class="label">' + soy.$$escapeHtml(opt_data.parentPage.title) + '</span></a></li></ul>' : '');
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.Pages.parentPage.soyTemplateName = 'Confluence.Templates.Sidebar.Pages.parentPage';
}


Confluence.Templates.Sidebar.Pages.renderChildren = function(opt_data, opt_ignored) {
  var output = '';
  var childList65 = opt_data.children;
  var childListLen65 = childList65.length;
  for (var childIndex65 = 0; childIndex65 < childListLen65; childIndex65++) {
    var childData65 = childList65[childIndex65];
    output += '<li class="child-item" data-page-id="' + soy.$$escapeHtml(childData65.pageId) + '"><span class="icon"></span><a href="' + soy.$$escapeHtml(childData65.url) + '" title="' + soy.$$escapeHtml(childData65.title) + '"><span class="label">' + soy.$$escapeHtml(childData65.title) + '</span></a></li>';
  }
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Sidebar.Pages.renderChildren.soyTemplateName = 'Confluence.Templates.Sidebar.Pages.renderChildren';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.editor-loader:background-loading-editor', location = 'jscripts/editor-loader.js' */
define("confluence-editor-loader/editor-loader","jquery confluence/legacy wrm/context-path confluence/dark-features confluence/api/logger confluence/meta confluence/template-renderer confluence/aui-overrides ajs wrm window".split(" "),function(d,h,n,o,e,i,p,q,f,r,s){var b,c={_listening:false,_queuedHandlers:[],_watchHandler:function(){h.Editor.UI.toggleWatchPage(false)},_unwatchHandler:function(){h.Editor.UI.toggleWatchPage(true)},_createQueueAdder:function(a){return function(){c._listening&&c._queuedHandlers.push(a)}},
bind:function(){f.bind("watchpage.pageoperation",this._createQueueAdder(this._watchHandler));f.bind("unwatchpage.pageoperation",this._createQueueAdder(this._unwatchHandler))},setListening:function(a){this._listening=a},applyHandlers:function(){for(var a=this._queuedHandlers.pop();a;){a();a=this._queuedHandlers.pop()}}};c.setListening(true);c.bind();var l=function(){var a=d("#editor-preload-container");a.length||(a=d('<div id="editor-preload-container" style="display: none;"></div>'));return a},g,
m={getPreloadContainer:l,getEditorPreloadMarkup:function(){if(g)return g;h.debugTime("confluence.editor.preload");var a=n()+"/plugins/editor-loader/editor.action";return g=d.get(a,{parentPageId:i.get("parent-page-id"),pageId:i.get("page-id"),spaceKey:i.get("space-key"),atl_after_login_redirect:s.location.pathname,timeout:m.loadingTimeout})},resourcesLoaded:function(){return b&&b.state()==="resolved"},loadingTimeout:12E3,isEditorActive:function(){var a=d("#editor-preload-container");return a.length&&
a.is(":visible")},load:function(a,c){function g(){var a=[o.isEnabled("frontend.editor.v4")?"wrc!editor-v4":"wrc!editor-v3","wrc!editor","wrc!macro-browser","wrc!fullpage-editor"];return r.require(a).fail(function(a){e.logError("Failed to load editor resources",a)})}var k;if(b){b.fail(function(){c?c.call(this,arguments):e.log("EditorLoader: loadGuard - previous load failed.")});b.done(function(){a?b.done(function(){setTimeout(a,0)}):e.log("EditorLoader: loadGuard - editor is already loaded.")});k=
true}else k=void 0;if(!k){b=new d.Deferred;a&&b.done(a);c&&b.fail(c);var f=l();d("body").append(f);var j=new d.Deferred;i.get("page-id")?this.getEditorPreloadMarkup().always(function(a,b,c){if(b==="success"||b==="notmodified"){f.append(a);a=p.renderTemplate("dynamic-editor-metadata");d("head").append(a);q.metaToParams();e.debug("EditorLoader: Finished loading the editor template.");j.resolve();h.debugTimeEnd("confluence.editor.preload")}else j.reject("Error loading the Editor template: "+c.status+
" - "+c.statusText)}):j.resolve();j.pipe(g).done(function(){e.debug("EditorLoader: Finished loading the editor.");b.resolve()}).fail(function(){b.reject(arguments)})}},getEditorForm:function(){if(this.isEditorActive()){var a=require("tinymce");return d(a.activeEditor.getContainer()).closest("form")}return null}};return m});require("confluence/module-exporter").exportModuleAsGlobal("confluence-editor-loader/editor-loader","AJS.Confluence.EditorLoader");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.editor-loader:background-loading-editor', location = 'jscripts/block-and-buffer-keys.js' */
define("confluence-editor-loader/block-and-buffer-keys",[],function(){return{block:function(e){var d=[],f=function(a){a.preventDefault();a.stopPropagation();var c=a.which;c||(c=a.charCode?a.charCode:a.keyCode);13!==c&&48>c||d.push(c);a.preventDefault()};e.keypress(f);return function(){e.unbind("keypress",f);for(var a="",c=0;c<d.length;c++){var b;b=d[c];65535<b?(b-=65536,b=String.fromCharCode(55296+(b>>10),56320+(b&1023))):b=String.fromCharCode(b);a+=b}return a}}}});
require("confluence/module-exporter").exportModuleAsGlobal("confluence-editor-loader/block-and-buffer-keys","AJS.Confluence.BlockAndBuffer");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.splitchunk.vendors--fc45e4dda1', location = 'aui.chunk.fbeb34172411fa3c260c--628726ce18a2dc2640a5.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.splitchunk.vendors--fc45e4dda1"],{"FJx+":function(i,n,s){"use strict";s.r(n)}}]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_jquery.ui.sortable', location = 'aui.chunk.f42656964f3e8eb3302b--c6bb4ef4f079172f7624.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["jquery.ui.sortable"],{Ckgl:function(u,i,n){"use strict";n.r(i);n("FJx+"),n("xTPO");i.default="jquery"}},[["Ckgl","runtime","aui.splitchunk.vendors--20a97d6a33","aui.splitchunk.vendors--d18e3cafa7","aui.splitchunk.vendors--db57146687","aui.splitchunk.vendors--51504ebf10","aui.splitchunk.vendors--351ae504d7","aui.splitchunk.vendors--6ce18a1d80","aui.splitchunk.vendors--fc45e4dda1","aui.splitchunk.0d131bcbf1"]]]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/linkbrowser-editor-adapter.js' */
AJS.$(function(){$("body").hasClass("with-space-sidebar")&&(Confluence=Confluence||{},Confluence.Editor=Confluence.Editor||{},AJS.Rte=AJS.Rte||{},AJS.Rte.BookmarkManager=AJS.Rte.BookmarkManager||{},AJS.Rte.BookmarkManager.storeBookmark=AJS.$.noop,AJS.Rte.BookmarkManager.restoreBookmark=AJS.$.noop)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/sidebar-ia.js' */
(function(E){Confluence.Sidebar={};AJS.toInit(function(b){function n(){AJS.trigger("sidebar.hide-overlays")}function l(a,c){b(document).tooltip(b.extend({gravity:"w",title:"data-tooltip",delayIn:500,delayOut:0,offset:5},c?c:{},{live:a}))}function v(){function a(){b(".tipsy").remove()}b(".acs-side-bar .quick-links-section").attr("data-collapsed-tooltip","Space shortcuts");b("#space-tools-menu-trigger").attr("data-collapsed-tooltip","Space tools");
"pages"==t.attr("data-tree-type")&&(b(".acs-side-bar .ia-secondary-container").attr("data-collapsed-tooltip","Child pages"),l(".collapsed .ia-secondary-container.tipsy-enabled",{title:"data-collapsed-tooltip"}));l(".expand-collapse-trigger");l(".ia-splitter-handle.tipsy-enabled");l(".collapsed .quick-links-section.tipsy-enabled, .collapsed .acs-nav-item \x3e a.tipsy-enabled, .collapsed #space-tools-menu-trigger.tipsy-enabled",{title:"data-collapsed-tooltip"});l(".configure-mode .acs-side-bar-space-info.tipsy-enabled",
{title:"data-configure-tooltip"});e.on("mousedown click scroll",n);b(window).on("scroll resize",n);AJS.bind("sidebar.hide-overlays",a);AJS.bind("sidebar.disable-tooltip",function(c,g){c=b(g).closest(".tipsy-enabled");if(1==c.size()){c.removeClass("tipsy-enabled").addClass("tipsy-disabled").attr("title","");if(c=c.data("tipsy"))c.hoverState="out";a()}});AJS.bind("sidebar.enable-all-tooltips",function(){b(".tipsy-disabled").removeClass("tipsy-disabled").addClass("tipsy-enabled")})}function y(a){"blogs"===
a?z(e,A):"pages"===a&&Confluence.Sidebar.Pages.installHandlers(e)}function A(a,c){var g=a.attr("data-group-type");a=a.attr("data-group-value");b.get(B+"/rest/ia/1.0/pagetree/blog/subtree",{spaceKey:C,groupType:g,groupValue:a}).done(c)}function z(a,c){a.delegate(".acs-tree-item \x3e .icon, .acs-tree-item \x3e .node-title","click",function(){var a=b(this).parent(),d=a.find("\x3e .icon");if(a.hasClass("opened"))a.children("ul").hide(),a.removeClass("opened").addClass("closed"),a.attr("aria-expanded",
!1),d.removeClass("aui-iconfont-expanded").addClass("aui-iconfont-collapsed");else if(a.hasClass("closed")){var e=a.children("ul");if(e.length)e.show();else{var f=b(Confluence.Templates.Sidebar.treeThrobber());a.append(f);c(a,function(c){c=b(Confluence.Templates.Sidebar.pagetreeList({pagetree:c,isSubtree:!0}));f.remove();c.appendTo(a)})}a.removeClass("closed").addClass("opened");a.attr("aria-expanded",!0);d.removeClass("aui-iconfont-collapsed").addClass("aui-iconfont-expanded")}})}function k(){var a=
u.offset().top,c=f.scrollTop(),g=f.scrollLeft();0>c||c>q.height()-f.height()||0>g||g>q.width()-f.width()||("fixed"!==b("#header").css("position")?d.css({top:Math.max(a-c,0)+"px",left:Math.min(-1*g,0)+"px"}):d.css({left:Math.min(-1*g,0)+"px"}))}function h(a){a=Math.max(a,55);a=Math.min(a,640);w.setItemQuietly("width",a);150>=a?(d.addClass("collapsed"),b(".expand-collapse-trigger").removeClass("aui-iconfont-chevron-double-left").addClass("aui-iconfont-chevron-double-right").attr("aria-expanded",!1),
r.attr("data-tooltip",Confluence.Sidebar.expandTooltip),AJS.trigger("sidebar.collapsed")):d.hasClass("collapsed")&&(d.removeClass("collapsed"),r.attr("data-tooltip",Confluence.Sidebar.collapseTooltip),b(".expand-collapse-trigger").removeClass("aui-iconfont-chevron-double-right").addClass("aui-iconfont-chevron-double-left").attr("aria-expanded",!0),AJS.trigger("sidebar.expanded"));d.width(a);x.eq(1).css("margin-left",a+"px");D.css("margin-left",d.outerWidth()+"px")}var f=b(window),q=b(document),m=
Math.min(285,f.width()/3),w=AJS.storageManager("confluence","sidebar"),B=AJS.contextPath(),C=AJS.Meta.get("space-key"),p=AJS.Meta.get("use-keyboard-shortcuts")?" "+"(\u2009[\u2009)":"";Confluence.Sidebar.collapseTooltip="Collapse sidebar"+p;Confluence.Sidebar.expandTooltip="Expand sidebar"+p;var x=b(".ia-splitter").children(),u=b(".ia-splitter-left");if(!(1>u.length)){var e=b(".acs-side-bar"),d=u.find(".ia-fixed-sidebar"),
r=b("\x3cdiv\x3e",{"class":"ia-splitter-handle tipsy-enabled","data-tooltip":Confluence.Sidebar.collapseTooltip}).appendTo(d);b("\x3cdiv\x3e",{"class":"ia-splitter-handle-highlight confluence-icon-grab-handle"}).appendTo(r);var t=b(".ia-secondary-container"),D=b("#footer, #studio-footer");Confluence.Sidebar.throbberDiv=function(){var a=b(Confluence.Templates.Sidebar.throbber()),c=a.find(".spinner"),d=Raphael.spinner(c[0],10,"#666");a.find(".throbber").bind("remove",function(){d()});return a};t.length&&
y(t.attr("data-tree-type"));f.scroll(k);f.resize(k);f.on("touchend",k);q.ready(k);AJS.bind("confluence.header-resized",k);b("#header-precursor img").load(k);Confluence.Sidebar.applyTooltip=l;v();AJS.bind("sidebar.exit-configure-mode",v);p=w.getItem("width")||m;h(150<p?p:55);(function(){q.on("mousewheel",".ia-scrollable-section",function(a,c){var d=b(this).scrollTop(),e=b(this).get(0).scrollHeight-b(this).innerHeight()-1;(0<c&&0>=d||0>c&&d>=e)&&a.preventDefault();a.stopPropagation()})})();d.css("visibility",
"visible");k();(function(){function a(){if(!Confluence.Sidebar.Configure.mode){var a=d.width();55<a?150>=a?(m=285,h(m)):(m=a,h(55)):h(m)}}var c=b("body"),e=!1,f=function(a){a.preventDefault();x.one("selectstart",function(a){a.preventDefault()});e=!1;c.on("mousemove.ia-splitter",function(a){Confluence.Sidebar.Configure.mode&&285>a.pageX||(h(a.pageX),e=!0)});c.one("mouseup mouseleave",function(){150>=d.width()?h(55):m=d.width();c.off("mousemove.ia-splitter")})};r.on("mousedown.ia-splitter",function(a){f(a);
n()}).click(function(){e?e=!1:a()});Confluence.Sidebar.toggle=a})();setTimeout(function(){Confluence.Sidebar.createFlyouts(e)},0);AJS.trigger("sidebar.finished-loading");f.one("pagetree-children-loaded",function(){var a=b(".plugin_pagetree_current");a.length&&(a=a.offset(),a.top>e.height()/2&&e.scrollTop(a.top-e.height()/3),a.left>e.width()/2&&e.scrollLeft(a.left-e.width()/2))});AJS.bind("sidebar.enter-configure-mode",function(){n();285>d.width()&&(Confluence.Sidebar.widthBeforeConfiguring=d.width(),
h(285));d.addClass("configure-mode")});AJS.bind("sidebar.exit-configure-mode",function(){n();Confluence.Sidebar.widthBeforeConfiguring&&(h(Confluence.Sidebar.widthBeforeConfiguring),Confluence.Sidebar.widthBeforeConfiguring=void 0);d.removeClass("configure-mode")})}})})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/configurable-nav.js' */
(function(d){AJS.Confluence.ConfigurableNav=AJS.RestfulTable.extend({initialize:function(b){var a=this;a.options=d.extend(!0,this._getDefaultOptions(b),b,{columns:[{id:"title"}]});a._events=a._events||AJS.RestfulTable.Events;a._event=a._event||AJS.RestfulTable.Events;a.classNames=AJS.RestfulTable.ClassNames;a.dataKeys=AJS.RestfulTable.DataKeys;a.$table=b.$el.addClass(this.classNames.RESTFUL_TABLE).addClass(this.classNames.ALLOW_HOVER).addClass("aui").addClass(a.classNames.LOADING);a.$table.prepend('\x3ccolgroup\x3e\x3ccol span\x3d"1" class\x3d"aui-restfultable-order"\x3e\x3ccol span\x3d"1"\x3e\x3ccol span\x3d"1" class\x3d"aui-restfultable-operations"\x3e\x3c/colgroup\x3e');
a.$tbody=d("\x3ctbody/\x3e");a._models=this._createCollection();a._rowClass=AJS.Confluence.ConfigurableNav.Row;a.editRows=[];a.enableReordering();a._models.bind("remove",function(b){d.each(a.getRows(),function(c,d){d.model===b&&(d.hasFocus()&&a._createRow&&a._createRow.trigger(a._event.FOCUS),a.removeRow(d))})});this.fetchInitialResources();Confluence.Sidebar.applyTooltip(".aui-iconfont-add-circle, .aui-iconfont-cross-circle, .aui-iconfont-plan-disabled",{gravity:"ne"})},enableReordering:function(){var b=
this;this.$tbody.sortable({handle:"."+this.classNames.DRAG_HANDLE,helper:function(a,c){a=c.clone(!0).addClass(b.classNames.MOVEABLE);a.children().each(function(a){d(this).width(c.children().eq(a).width())});return a},start:function(a,c){var e=c.placeholder.find("td");c.item.addClass(b.classNames.MOVEABLE).children().each(function(a){d(this).width(e.eq(a).width())});c.placeholder.html('\x3ctd colspan\x3d"'+b.getColumnCount()+'"\x3e\x26nbsp;\x3c/td\x3e').css("visibility","visible");b.getRowFromElement(c.item[0]).trigger(b._event.MODAL)},
stop:function(a,c){jQuery(c.item[0]).is(":visible")&&(c.item.removeClass(b.classNames.MOVEABLE).children().attr("style",""),c.placeholder.removeClass(b.classNames.ROW),b.getRowFromElement(c.item[0]).trigger(b._event.MODELESS))},update:function(a,c){a={};var e=b.getRowFromElement(c.item[0]);if(e){var f=function(a,b,c){d(b).trigger(a,c);d(b).trigger(a,c);b.id&&d(b).trigger(b.id+"-"+a,c)};c=b.options.reverseOrder?c.item.next():c.item.prev();c.length?(c=b.getRowFromElement(c).model,a.after=c.url()):a.position=
"First";a.spaceKey=AJS.Meta.get("space-key");d.ajax({url:e.model.url()+"/move",type:"POST",dataType:"json",contentType:"application/json",data:JSON.stringify(a),complete:function(){e.hideLoading()},success:function(a){f(AJS.RestfulTable.Events.REORDER_SUCCESS,b,[a])},error:function(a){var c=d.parseJSON(a.responseText||a.data);f(AJS.RestfulTable.Events.SERVER_ERROR,b,[c,a])}});e.showLoading()}},axis:"y",delay:0,containment:"document",cursor:"move",scroll:!0,zIndex:8E3});this.$tbody.bind("selectstart mousedown",
function(a){return!d(a.target).is("."+b.classNames.DRAG_HANDLE)})}});AJS.Confluence.ConfigurableNav.ReadView=AJS.RestfulTable.CustomReadView.extend({render:function(b){return _.template('\x3cspan class\x3d"acs-nav-item-link" title\x3d"\x3c%\x3dtitle%\x3e"\x3e\x3cspan class\x3d"'+("pinned_attachment"==b.styleClass?"aui-icon aui-icon-small aui-iconfont-attachment":"icon")+'"\x3e\x3c/span\x3e\x3cspan class\x3d"acs-nav-item-label"\x3e\x3c%\x3dtitle%\x3e\x3c/span\x3e\x3c/span\x3e')({title:AJS.escapeHtml(b.title)})}});
AJS.Confluence.ConfigurableNav.Row=AJS.RestfulTable.Row.extend({render:function(){var b=this,a=this.model.toJSON(),c=d("\x3ctd class\x3d'aui-restfultable-operations' /\x3e").append(this.renderOperations(a.canHide,a.hidden)),e=d('\x3ctd class\x3d"'+this.classNames.ORDER+'"/\x3e').append(this.renderDragHandle());b._event=b._event||b._events;b.$el.attr("data-id",this.model.id);b.$el.append(e);d.each(b.columns,function(c,e){c=d("\x3ctd /\x3e");var g=a[e.id];g&&b.$el.attr("data-"+e.id,g);e=(new AJS.Confluence.ConfigurableNav.ReadView).render(a);
c.append(e);b.$el.append(c)});b.$el.append(c);a.canHide&&a.hidden&&b.$el.addClass("hidden-link");b.$el.addClass(this.classNames.ROW+" "+b.classNames.READ_ONLY+" acs-nav-item "+a.styleClass);b.trigger(this._event.RENDER,this.$el,a);b.$el.trigger(this._event.CONTENT_REFRESHED,[b.$el]);return b},renderOperations:function(b,a){var c=this,e=d('\x3ca href\x3d"#" class\x3d"aui-icon aui-icon-small"/\x3e');if(b){var f=function(a){a.hasClass("aui-iconfont-plan-disabled")?a.attr("data-tooltip","Hide link"):
a.attr("data-tooltip","Show link")};e.addClass(a?"aui-iconfont-add-circle show-link":"aui-iconfont-plan-disabled hide-link").click(function(b){b.preventDefault();d.ajax({url:c.model.url()+(a?"/show":"/hide"),type:"POST",dataType:"json",contentType:"application/json",data:JSON.stringify({spaceKey:AJS.Meta.get("space-key")})}).done(function(){e.closest(".acs-nav-item").toggleClass("hidden-link");e.toggleClass("aui-iconfont-plan-disabled").toggleClass("aui-iconfont-add-circle");
e.toggleClass("hide-link").toggleClass("show-link");f(e)})});f(e)}else e.addClass("aui-iconfont-cross-circle delete-link tipsy-enabled").click(function(a){a.preventDefault();"hide"!=d(".acs-nav").data("quick-links-state")&&(AJS.trigger("sidebar.disable-tooltip",this),c.destroy())}).attr("data-tooltip","Remove link");return e},destroy:function(){this.model.destroy({data:{spaceKey:AJS.Meta.get("space-key")}})}})})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/sidebar-pages.js' */
(function(a){var d=AJS.Meta.get("context-path");Confluence.Sidebar.Pages={installHandlers:function(c){c.find(".more-children-link").click(function(b){b.preventDefault();c.find("ul.more-children").show();a(this).hide()})},quickLinksContent:function(){return(new a.Deferred).resolve(a(".acs-side-bar .quick-links-wrapper").html())},childPageCollapsedContent:function(){var c=a(".acs-side-bar .ia-secondary-header"),b=a(".acs-side-bar .ia-secondary-parent-content"),d=a(".acs-side-bar .ia-secondary-current-content"),
e=a(".acs-side-bar .ia-secondary-content");return(new a.Deferred).resolve(a("\x3cdiv\x3e").append(a("\x3cdiv\x3e").addClass("acs-side-bar-flyout-wiki-wrapper").append(c.clone()).append(b.clone()).append(d.clone()).append(e.clone())).html())},pageTreeCollapsedContent:function(){var c=a(".page-tree-flyout-content");return 0==c.length?a.ajax({url:d+"/rest/ia/1.0/space/childPagesContextualNav",data:{pageId:AJS.Meta.get("page-id")}}).pipe(function(b){b=a("\x3cdiv\x3e").addClass("acs-side-bar-flyout-wiki-wrapper").append(Confluence.Templates.Sidebar.Pages.renderPageContextualNav({pageContextualNav:b,
hasCreatePermission:a(".acs-nav").data("has-create-permission")}));a("body").append(a("\x3cdiv\x3e").addClass("page-tree-flyout-content hidden").append(b.clone()));return b}):(new a.Deferred).resolve(c.html())}}})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/sidebar-links.js' */
AJS.$(function(){function u(b){var e=a(".space-information-container .favourite-space-icon");e.addClass("disabled");if("add"===b){var l="PUT";var d="confluence.space-sidebar.favourite.click"}else l="DELETE",d="confluence.space-sidebar.favourite-remove.click";a.ajax({url:f+"/rest/experimental/relation/user/current/favourite/toSpace/"+h,type:l,dataType:"json",contentType:"application/json",data:JSON.stringify({spaceKey:h})}).done(function(){var c=a("#space-favourite-add"),n=a("#space-favourite-remove");
"add"===b?(c.prop("data-favourited","true"),c.attr("aria-pressed","true"),n.attr("aria-pressed","false"),n.show(),c.hide()):(c.prop("data-favourited","false"),c.attr("aria-pressed","false"),n.attr("aria-pressed","true"),c.show(),n.hide())}).fail(function(a){var b=require("confluence/message-controller");b.showError(b.parseError(a),b.Location.FLAG)}).always(function(){e.removeClass("disabled")});AJS.trigger("analyticsEvent",{name:d})}function w(){AJS.trigger("sidebar-before-enter-configure-mode");
var b=a(".custom-sidebar-content"),e=b.length;if(e){var l=f+"/spaces/custompagecontent.action?key\x3d"+h;var m=b.children().html()}p=d.data("nav-type");Confluence.Sidebar.Configure.mode=!0;v=a(".acs-nav-sections .acs-nav-item.current-item").data("collector-key");g=Confluence.Sidebar.throbberDiv();b=a(".acs-nav-sections");g.height(b.height());q.hide();b.replaceWith(g);e=Confluence.Templates.Sidebar.configure({pageTree:"page-tree"===p,quickLinksState:d.data("quick-links-state"),pageTreeState:d.data("page-tree-state"),
hasSidebarCustomisation:e,customContentAdminLink:l,sidebarCustomisation:m,accessMode:AJS.Meta.get("access-mode")});k=a(e).hide();g.after(k);x();y();Confluence.Sidebar.Configure.MainLinks=new AJS.Confluence.ConfigurableNav({$el:a("#acs-nav-list-main"),resources:{all:f+"/rest/ia/1.0/link/main?spaceKey\x3d"+h+"\x26includeHidden\x3dtrue",self:f+"/rest/ia/1.0/link"}});Confluence.Sidebar.Configure.QuickLinks=new AJS.Confluence.ConfigurableNav({$el:a("#acs-nav-list-quick"),resources:{all:f+"/rest/ia/1.0/link/quick?spaceKey\x3d"+
h,self:f+"/rest/ia/1.0/link"},reverseOrder:!0});var c={};a("#acs-nav-list-main").one(AJS.RestfulTable.Events.INITIALIZED,function(){c.main=!0;c.quick&&(g.replaceWith(k),k.show())});a("#acs-nav-list-quick").one(AJS.RestfulTable.Events.INITIALIZED,function(){c.quick=!0;c.main&&(g.replaceWith(k),k.show())});a(".acs-nav-type").change(function(b){r("nav-type",a(this).val(),function(a){q.data("tree-type",a);d.data("nav-type",a)})});a(".acs-done-link").click(function(b){b.preventDefault();a(".acs-done-link").attr("aria-disabled",
!0).prop("disabled",!0);z()});a(".quick-links-header a").click(function(a){a.preventDefault();a="hide"===d.data("quick-links-state")?"show":"hide";r("quick-links-state",a,A)});a(".page-tree-header a").click(function(b){b.preventDefault();b="hide"===d.data("page-tree-state")?"show":"hide";var c=a(".acs-nav-type");"show"===b?c.attr("disabled",!1):c.attr("disabled",!0);r("page-tree-state",b,B)});AJS.$(".acs-side-bar-space-info").on("click.configurelogo",C);AJS.trigger("sidebar.enter-configure-mode")}
function C(a){var b=AJS.$(".acs-side-bar-space-info \x3e .flyout-handle");b.addClass("loading").spin();WRM.require("wr!com.atlassian.confluence.plugins.confluence-space-ia:avatar-picker",function(){AJS.trigger("deferred.spaceia.open.configure.space")}).always(function(){b.removeClass("loading").spinStop()});a.preventDefault()}function r(b,e,d){a.ajax({url:f+"/rest/ia/1.0/space/option",type:"POST",dataType:"json",contentType:"application/json",data:JSON.stringify({spaceKey:h,option:b,value:e}),success:function(a){d(e)}})}
function z(){if(p!==d.data("nav-type")||!0===t)location.reload(),t=!1;else{var b=a(".custom-sidebar-content"),e=b.length,l;e&&(l=b.children().html());g=Confluence.Sidebar.throbberDiv();g.height(k.height());k.replaceWith(g);Confluence.Sidebar.Configure.MainLinks.remove();Confluence.Sidebar.Configure.MainLinks.unbind();Confluence.Sidebar.Configure.QuickLinks.remove();Confluence.Sidebar.Configure.QuickLinks.unbind();a(document).unbind(AJS.RestfulTable.Events.INITIALIZED);a(document).unbind(AJS.RestfulTable.Events.ROW_ADDED);
a(document).unbind(AJS.RestfulTable.Events.ROW_REMOVED);q.show().css("display","");var m=function(){var b=a(Confluence.Templates.Sidebar.renderLinks({mainLinks:c.main,quickLinks:c.quick.reverse(),advancedLinks:c.advanced,hasConfigurePermission:!0,collectorToHighlight:v,quickLinksState:d.data("quick-links-state"),hasSidebarCustomisation:e,sidebarCustomisation:l}));g.replaceWith(b);Confluence.Sidebar.Configure.mode=!1;AJS.trigger("sidebar.exit-configure-mode")},c={};a.get(f+"/rest/ia/1.0/link/main",
{spaceKey:h,includeHidden:!1}).done(function(a){c.main=a;c.quick&&c.advanced&&m()});a.get(f+"/rest/ia/1.0/link/quick",{spaceKey:h}).done(function(a){c.quick=a;c.main&&c.advanced&&m()});a.get(f+"/rest/ia/1.0/link/advanced",{spaceKey:h}).done(function(a){c.advanced=a;c.main&&c.quick&&m()});Confluence.Sidebar.Configure.Logo&&Confluence.Sidebar.Configure.Logo.unbind()}}function x(){WRM.require("wr!com.atlassian.confluence.plugins.confluence-space-ia:link-dialog",function(){var b=new a.Deferred;a(".acs-add-link").click(function(e){b.done(function(){e.preventDefault();
"hide"!==d.data("quick-links-state")&&(Confluence.Sidebar.LinkAdapter.hijackLinkBrowser(),Confluence.Editor.LinkBrowser.open(),a("#recentlyviewed-panel-id").click())})}).addClass("acs-add-link-ready");AJS.Meta.get("space-key")?AJS.Confluence.EditorLoader.load(function(){b.resolve()},function(){AJS.log("Attempted to load editor for space ia side bar. Loading the editor failed.");b.reject()}):b.resolve()})}function y(){var b=a("#acs-nav-list-quick"),e=a(".acs-nav-sections .tip").hide(),d=function(){Confluence.Sidebar.Configure.QuickLinks.isEmpty()?
(b.hide(),e.show()):(e.hide(),b.show())};a(document).bind(AJS.RestfulTable.Events.INITIALIZED,d);a(document).bind(AJS.RestfulTable.Events.ROW_ADDED,d);a(document).bind(AJS.RestfulTable.Events.ROW_REMOVED,d)}function A(b){d.data("quick-links-state",b);"hide"===b?(a(".acs-nav-list-quick-section").addClass("hidden-section"),a(".quick-links-header a").removeClass("aui-iconfont-plan-disabled").addClass("aui-iconfont-add-circle").attr("aria-disabled",!0)):(a(".acs-nav-list-quick-section").removeClass("hidden-section"),
a(".quick-links-header a").removeClass("aui-iconfont-add-circle").addClass("aui-iconfont-plan-disabled").attr("aria-disabled",!1))}function B(b){d.data("page-tree-state",b);t=!0;"hide"===b?(a(".acs-nav-list-page-tree-section").addClass("hidden-section"),a(".page-tree-header a").removeClass("aui-iconfont-plan-disabled").addClass("aui-iconfont-add-circle").attr("aria-disabled",!0)):(a(".acs-nav-list-page-tree-section").removeClass("hidden-section"),a(".page-tree-header a").removeClass("aui-iconfont-add-circle").addClass("aui-iconfont-plan-disabled").attr("aria-disabled",
!1))}Confluence.Sidebar.Configure={mode:!1};var a=AJS.$,f=AJS.Meta.get("context-path"),h=AJS.Meta.get("space-key"),q=a(".acs-side-bar").find(".ia-secondary-container"),g,k,v,p,d=a(".acs-nav"),t=!1;a.ajaxSetup({cache:!1});a("body").on("click","#acs-configure-link, a.configure-sidebar",function(a){a.preventDefault();w()});a("#space-favourite-remove").on("click",function(){u("remove")});a("#space-favourite-add").on("click",function(){u("add")})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/sidebar-space-tools.js' */
define("confluence-space-ia/sidebar-space-tools",["ajs","jquery","confluence/legacy"],function(b,a,c){return function(){var g=[],h=[],f=0===b.version.indexOf("5.7");if(!f){var k=a("#space-tools-menu-trigger");var e=a("#space-tools-menu")}f?a("#space-tools-web-items").children("div").each(function(){g.push({label:a(this).data("label"),href:a(this).data("href")})}):(b.warn("Remove legacy sidebar code when upgrade to AUI 5.8+"),e.on({"aui-dropdown2-show":function(){b.trigger("sidebar.disable-tooltip",
k)},"aui-dropdown2-hide":function(){b.trigger("sidebar.enable-all-tooltips")}}),a("#space-tools-web-items").children("div").each(function(){g.push({text:a(this).data("label"),href:a(this).data("href")})}));f?a("#space-tools-menu-additional-items").children("div").each(function(){h.push({className:a(this).data("class"),label:a(this).data("label"),href:a(this).data("href")})}):a("#space-tools-menu-additional-items").children("div").each(function(){h.push({extraClasses:a(this).data("class"),text:a(this).data("label"),
href:a(this).data("href")})});if(f){var m={hideDelay:null,width:170,displayShadow:!1,calculatePositions:function(a,b){b=b.target.offset();return{popupCss:{top:b.top-a.height(),left:b.left},arrowCss:{display:"none"}}},hideCallback:function(){b.trigger("sidebar.enable-all-tooltips")}};var l=b.InlineDialog(a("#space-tools-menu-trigger"),"space-tools",function(d,e,k){d.html(c.Templates.Sidebar.spaceToolsMenu({spaceToolLinks:g,spaceLinks:h,isAuiFiveSeven:f}));a(e).one("click",function(b){a("#inline-dialog-space-tools").is(":visible")&&
setTimeout(function(){l.hide()},0)});b.trigger("sidebar.disable-tooltip",e);b.trigger("sidebar.spacetools-loaded");k();return!1},m);l.addClass("aui-dropdown2 aui-style-default space-tools-dropdown");b.bind("sidebar.hide-overlays",l.hide)}else e.html(c.Templates.Sidebar.spaceToolsMenu({spaceToolLinks:g,spaceLinks:h})),b&&b.Confluence&&b.Confluence.Analytics&&b.Confluence.Analytics.setAnalyticsSource&&b.Confluence.Analytics.setAnalyticsSource(e.find("a:not(.configure-sidebar)"),"spacetools"),b.bind("sidebar.hide-overlays",
function(){e.filter('[aria-hidden\x3d"false"]').length&&a("#space-tools-menu-trigger").trigger("aui-button-invoke")});var d=a(".expand-collapse-trigger");d.click(function(a){a.preventDefault();c.Sidebar.toggle()});d.attr("data-tooltip",a(".ia-fixed-sidebar").hasClass("collapsed")?c.Sidebar.expandTooltip:c.Sidebar.collapseTooltip);d.attr("aria-label",a(".ia-fixed-sidebar").hasClass("collapsed")?c.Sidebar.expandTooltip:c.Sidebar.collapseTooltip);b.bind("sidebar.collapsed",function(){d.attr("data-tooltip",
c.Sidebar.expandTooltip);d.attr("aria-label",c.Sidebar.expandTooltip)});b.bind("sidebar.expanded",function(){d.attr("data-tooltip",c.Sidebar.collapseTooltip);d.attr("aria-label",c.Sidebar.collapseTooltip)})}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/sidebar-space-tools-require.js' */
require(["confluence-space-ia/sidebar-space-tools"],function(a){AJS.toInit(function(){a()})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/sidebar-flyouts.js' */
(function(c){function d(a,b,g,d){var f=AJS.InlineDialog(a,g,function(a,h,e){c(a).addClass("acs-side-bar-flyout ia-scrollable-section");c(a).empty().append(Confluence.Sidebar.throbberDiv());b().done(function(b){c(a).html(b)});AJS.trigger("sidebar.flyout-triggered",d);e();c(h).one("click",function(a){c("#inline-dialog-"+g).is(":visible")&&setTimeout(function(){f.hide()},0)});AJS.trigger("sidebar.disable-tooltip",h)},{gravity:"w",calculatePositions:e,useLiveEvents:!0,hideDelay:null,hideCallback:function(){AJS.trigger("sidebar.enable-all-tooltips")}});
AJS.bind("sidebar.hide-overlays",f.hide);return f}function e(a,b,d,e){a=b.target.offset();d=b.target.width();b=b.target.height();b={top:a.top+b/2-15,left:a.left+d+5,right:"auto"};a=c(window);b.maxHeight=a.height()+a.scrollTop()-b.top-20;return{popupCss:b,arrowCss:{top:9},gravity:"w"}}Confluence.Sidebar.createFlyouts=function(a){d(c(".collapsed .quick-links-section"),Confluence.Sidebar.Pages.quickLinksContent,"sidebar-quick-links-flyout",{flyout:"quick-links"});a=a.find(".ia-secondary-container");
a.length&&"pages"==a.attr("data-tree-type")&&d(c(".collapsed .ia-secondary-header-title.wiki"),Confluence.Sidebar.Pages.childPageCollapsedContent,"sidebar-children-flyout",{flyout:"children"});a.length&&"page-tree"==a.attr("data-tree-type")&&d(c(".collapsed .ia-secondary-header-title.page-tree"),Confluence.Sidebar.Pages.pageTreeCollapsedContent,"sidebar-page-tree-flyout",{flyout:"pagetree"})}})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/external/jquery.mousewheel.min.js' */
/*! Copyright (c) 2011 Brandon Aaron (http://brandonaaron.net)
 * Licensed under the MIT License (LICENSE.txt).
 *
 * Thanks to: http://adomas.org/javascript-mouse-wheel/ for some pointers.
 * Thanks to: Mathias Bank(http://www.mathias-bank.de) for a scope bug fix.
 * Thanks to: Seamus Leahy for adding deltaX and deltaY
 *
 * Version: 3.0.6
 * 
 * Requires: 1.2.2+
 */
(function(a){function d(b){var c=b||window.event,d=[].slice.call(arguments,1),e=0,f=!0,g=0,h=0;return b=a.event.fix(c),b.type="mousewheel",c.wheelDelta&&(e=c.wheelDelta/120),c.detail&&(e=-c.detail/3),h=e,c.axis!==undefined&&c.axis===c.HORIZONTAL_AXIS&&(h=0,g=-1*e),c.wheelDeltaY!==undefined&&(h=c.wheelDeltaY/120),c.wheelDeltaX!==undefined&&(g=-1*c.wheelDeltaX/120),d.unshift(b,e,g,h),(a.event.dispatch||a.event.handle).apply(this,d)}var b=["DOMMouseScroll","mousewheel"];if(a.event.fixHooks)for(var c=b.length;c;)a.event.fixHooks[b[--c]]=a.event.mouseHooks;a.event.special.mousewheel={setup:function(){if(this.addEventListener)for(var a=b.length;a;)this.addEventListener(b[--a],d,!1);else this.onmousewheel=d},teardown:function(){if(this.removeEventListener)for(var a=b.length;a;)this.removeEventListener(b[--a],d,!1);else this.onmousewheel=null}},a.fn.extend({mousewheel:function(a){return a?this.bind("mousewheel",a):this.trigger("mousewheel")},unmousewheel:function(a){return this.unbind("mousewheel",a)}})})(jQuery);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacesidebar', location = 'js/sidebar-logo.js' */
AJS.toInit(function(a){Confluence.Sidebar.Configure.Logo={};var g=a(".acs-side-bar-space-info div.name a"),d,f,h=function(e){d||(d=new e({onCrop:function(e,c){a.ajax({type:"POST",dataType:"json",contentType:"application/json",data:JSON.stringify({spaceKey:AJS.Meta.get("space-key"),spaceName:c,logoDataURI:e}),url:AJS.Meta.get("context-path")+"/rest/ia/1.0/space/setLogo",success:function(b){var c=b.logoDownloadPath;a(".space-logo .avatar-img").attr("src",AJS.Meta.get("context-path")+c);b=b.name;g.attr("title",
b).text(b);d.hide()},error:function(b){d.setMessage("An error occurred while updating space details");d._removeSaveImageLoadingIcon()}})}}));d.show(a(".acs-side-bar-space-info div.name a").text());return!1},k=function(){function e(c,b,a,d){c=b.target.offset();a=b.target.width();b=b.target.height();return{popupCss:{top:c.top+b/2-15,left:c.left+a+5,right:"auto"},arrowCss:{top:9},gravity:"w"}}var d=function(c,b,d){a(c).addClass("acs-side-bar-flyout");a(c).empty();c.html(AJS.template.load("logo-config-content"));
c.unbind("mouseover mouseout");AJS.trigger("sidebar.disable-tooltip",b);d()};f||(f=AJS.InlineDialog(a(".acs-side-bar-space-info"),"space-logo-config",d,{gravity:"w",calculatePositions:e,useLiveEvents:!0,hideCallback:function(){AJS.trigger("sidebar.enable-all-tooltips")},hideDelay:null,noBind:!0,width:635}));f.show()};AJS.bind("sidebar-before-enter-configure-mode",function(){AJS.bind("deferred.spaceia.open.configure.space",function(a){AJS.Meta.get("space-key")=="~"+AJS.Meta.get("remote-user")?k():
require(["confluence-space-ia/avatar-picker/avatar-picker-dialog"],h);a.preventDefault();return!1})});Confluence.Sidebar.Configure.Logo.unbind=function(){a("#inline-dialog-space-logo-config .cancel").click();a(".acs-side-bar-space-info").off("click.configurelogo");AJS.unbind("deferred.spaceia.open.configure.space")}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-fixed-headers:utils', location = '/js/utils/dom.js' */
define("confluence/fh/utils/dom",["jquery","ajs","exports"],function(d,p,e){function g(a){if(!g.done||a)l=0!==d("#main .aui-page-panel-outer-content").length?d("#main .aui-page-panel-outer-content"):d("#main"),q=d("#header"),h=d("#main-header"),m=!1,r=d("#main-header #title-text"),g.done=!0}function t(){g();var a=parseInt(h.css("top"))+h.outerHeight();isNaN(a)||a===t.lastValue||(t.lastValue=a,p.trigger("sticky-table-headers.change.options",{fixedOffset:a,cacheHeaderHeight:!0}))}function k(a,b,c){if(c){k[c]=
k[c]||{};if(b===k[c].lastValue)return;k[c].lastValue=b}a.css({transform:"translateY("+b+"px)","-webkit-transform":"translateY("+b+"px)","-ms-transform":"translateY("+b+"px)"})}function u(a,b,c){var d={};c.forEach(function(b){d[b]=a.css(b)});b.css(d)}function w(a){g();if(f){var b=h.height(),c=a?b-r.height():b+parseInt(r.css("padding-top"));n||(n=f.find("h1"));n.css({paddingTop:c});a=a?b:n.outerHeight();f.height(a)}}var f,l,q,h,r,n,m;var x=["margin-top","margin-right","margin-left","margin-bottom"];
var y=["padding-top","padding-right","padding-left","padding-bottom"];var v=["font-family","font-size","font-style","font-weight"];e.forceInitialize=function(){g(!0)};e.addClassToPage=function(){var a=p.Meta.get("content-type");"page"!==a&&"blogpost"!==a||p.bind("init.rte",function(){d(tinymce.activeEditor.getWin().document).find("body#tinymce").addClass("page-edit")})};e.adjustMainHeaderSize=function(){g();h.css({width:l.outerWidth()-(parseInt(l.css("padding-left"))+parseInt(l.css("padding-right")))});
var a=!q.hasClass("fixed-header");w(a)};e.updateTableStickyHeaderOption=t;e.translateVertical=k;e.createMainHeaderPlaceHolder=function(){g();if(f)return f;f=d('\x3cdiv id\x3d"main-header-placeholder"\x3e\x3c/div\x3e');var a=d("#title-text"),b=a.find("a"),c=a.clone().removeAttr("id"),e=c.find("a");u(h,f,x.concat(y));f.css({height:h.outerHeight()});u(a,c,v.concat("text-decoration","letter-spacing","text-align","padding-right","padding-bottom","padding-left","margin-right","margin-bottom","margin-left"));
c.css({paddingTop:parseInt(a.css("padding-top"))+Math.round(a.offset().top-h.offset().top)});u(b,e,v.concat(x,y,v,"color","text-decoration","letter-spacing","text-align"));f.append(c);return f};e.removeMainHeaderPlaceHolder=function(){f&&f.remove()};e.scrollIfAnchor=function(a){var b;if(b="object"===typeof a)b=a,b=d.contains(document.querySelector("#header"),b.target);if(!b&&(a="string"===typeof a?a:d(this).attr("href"))){g();b=q.outerHeight();var c=d(a.replace(/(!|"|\$|%|&|'|\(|\)|\*|\+|,|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|`|\{|\||\}|~)/g,
"\\$1"));if(c.length&&(m=!0,window.scrollTo(0,c.offset().top-b),history&&"pushState"in history))return history.pushState({},document.title,window.location.pathname+window.location.search+a),!1}};e.isScrollingToAnchorLink=function(){return m};e.unsetScrollingToAnchorLinkFlag=function(){m=!1};e.adjustMainHeaderPlaceHolderSize=w});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-fixed-headers:handlers', location = '/js/handlers/page-view-handler.js' */
define("confluence/fh/handlers/page-view-handler",["confluence/fh/utils/dom","jquery","ajs","exports"],function(k,l,B,b){function q(b){if(!q.done||b)p=l("#header"),e=l("#main .aui-page-panel-outer-content"),0===e.length&&(e=l("#main")),a=l("#main-header"),l("#action-menu-link"),C=k.createMainHeaderPlaceHolder(),z=l(".ia-splitter-left .ia-fixed-sidebar"),f=a.prop("style"),D=e.prop("style"),E=Math.round(p.offset().top),d=p.height(),F=Math.round(a.offset().top)-d-(parseInt(e.css("padding-top"))+parseInt(a.css("margin-top"))),
q.done=!0}function G(){m&&0!==g&&(g=0,a.addClass("overlay-header"),a.css({top:d+"px",zIndex:100}),k.updateTableStickyHeaderOption())}var p,e,a,C,z,f,D,E,d,F,u=!1,m=!1,w=!1,A=0,g=0,r=0,x="by-other",H=function(){q();var b=k.isScrollingToAnchorLink();k.unsetScrollingToAnchorLinkFlag();var n=l(window).scrollTop(),c=n>E;c&&parseInt(z.css("top"))<d&&z.css({top:d+"px"});!u&&c?(u=!0,p.addClass("fixed-header"),C.css("padding-top",p.height())):c||(p.removeClass("fixed-header"),D.marginTop="",u=!1);c=n>F;!m&&
c?(m=!0,a.find("#title-text").hide(),a.css({position:"fixed",width:e.outerWidth()-(parseInt(e.css("padding-left"))+parseInt(e.css("padding-right"))),right:0,top:d+"px",marginTop:0,paddingTop:parseInt(e.css("padding-top"))+parseInt(a.css("margin-top")),paddingBottom:parseInt(e.css("padding-top"))+parseInt(a.css("margin-top")),paddingLeft:e.css("padding-left"),paddingRight:e.css("padding-right"),zIndex:100}),a.before(k.createMainHeaderPlaceHolder()),window.scrollTo(0,n)):c||(f.position="",f.width="",
f.right="",f.top="",f.marginTop="",f.paddingTop="",f.paddingBottom="",f.paddingLeft="",f.paddingRight="",f.zIndex="",a.removeClass("overlay-header"),a.find("#title-text").show(),w=m=!1,k.removeMainHeaderPlaceHolder());c=a.outerHeight();var t=A-n,y=parseInt(a.css("top"))+t;if(0>=n){g=r=0;var h=d}else if(b){r=0;g=c;h=d-c;var v=100}else if(m&&0<t){if(y>d)r=d,g=0,h=d,v=100;else if(r+=Math.abs(t),50<=r||n<=c)g-=Math.abs(t),h=y,v=100;n>c+50&&!w&&(a.addClass("overlay-header"),w=!0);x="by-scroll"}else m&&
0>t&&(y<d-c?(r=0,g=c,h=d-c,v=100):(g+=Math.abs(t),h=y));void 0!==v&&void 0!==h?a.css({top:h+"px",zIndex:v}):void 0!==h&&a.css({top:h+"px"});k.updateTableStickyHeaderOption();A=n};b.forceInitialize=function(){q(!0)};b.onScrollHandler=H;b.onHoverActionMenuLinkHandler=function(){q();G();x="by-hover"};b.onClickEditPageLinkHandler=function(){q();G();B.bind("quick-edit.viewport.saved",function(){p.removeAttr("style");k.removeMainHeaderPlaceHolder()});l(window).off("scroll",H).off("resize.confluence-fixed-headers-responsive");
B.trigger("analyticsEvent",{name:"view.edit.transition.edit.button.clicked",data:{method:x}})};b.isKeepDotDotDotButton=!1;b.__getShowMainHeaderBy=function(){return x};b.__getFixedHeader=function(){return u};b.__getFixedMainHeader=function(){return m};b.__setFixedHeader=function(a){u=a};b.__setFixedMainHeader=function(a){m=a};b.__setMainHeaderOverlay=function(a){w=a};b.__setScrollPosition=function(a){A=a};b.__setMovingDistance=function(a){g=a}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-fixed-headers:confluence-fixed-headers-view-content-resources', location = '/js/confluence-fixed-headers.js' */
require(["confluence/fh/utils/dom","confluence/fh/handlers/page-view-handler","ajs","jquery"],function(c,d,e,a){e.toInit(function(){if(e.Meta.get("content-type")&&!(0<a("#main-content").find("style").length||a("body").hasClass("theme-documentation"))){var k=a("#header"),l=a("#main-header"),h=a("#main"),f=a("#content"),m=a("#editPageLink"),n=a("#action-menu-link"),p=["inline-dialog-notifications-miniview","inline-dialog-confluence-watch","inline-dialog-shareContentPopup","inline-dialog-ap-inline-dialog-content-confstats-connect-dev__confstats-live-watcher",
"inline-dialog-jira-metadata-dialog"];(new MutationObserver(function(a){c.adjustMainHeaderSize()})).observe(h[0],{attributes:!0,attributeFilter:["style"]});var g=new MutationObserver(function(a){"0px"===f.css("padding-right")&&f.css({paddingRight:"28px"})});g.observe(f[0],{attributes:!0,attributeFilter:["style"]});a(window).on("resize.confluence-fixed-headers-responsive",e.debounce(c.adjustMainHeaderSize,100)).on("scroll",d.onScrollHandler).scroll(e.debounceImmediate(function(){var b=a('.aui-dropdown2[aria-hidden\x3d"false"]');
b.length&&a('.aui-dropdown2-trigger[aria-owns\x3d"'+b.attr("id")+'"]').trigger("aui-button-invoke");b=a(".aui-inline-dialog:visible");b.length&&(b=b.filter(function(){var b=a(this),c=b.attr("id");return b.hasClass("non-persistence-dialog")||-1!==p.indexOf(c)}),b.css({display:"none"}),k.find(".aui-button.active, .aui-nav-imagelink.active").removeClass("active"),l.find(".aui-button.active, .aui-nav-imagelink.active").removeClass("active"),document.activeElement.blur())},250));d.isKeepDotDotDotButton&&
(h.addClass("floating-action-menu"),n.on("hover",d.onHoverActionMenuLinkHandler));m.click(function(){g&&g.disconnect();d.onClickEditPageLinkHandler()});setTimeout(function(){c.scrollIfAnchor(window.location.hash)});a("body").on("click",'a[href^\x3d"#"]',c.scrollIfAnchor)}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-like:content-like-resources', location = '/com/atlassian/confluence/plugins/like/js/like-namespace.js' */
define("confluence-like/like-namespace",["confluence/legacy"],function(a){return a.Likes||{}});require("confluence/module-exporter").exportModuleAsGlobal("confluence-like/like-namespace","Confluence.Likes");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-like:content-like-resources', location = '/com/atlassian/confluence/plugins/like/js/like.js' */
define("confluence-like/like",["ajs","confluence/legacy","jquery","confluence/meta","confluence/message-controller"],function(e,c,d,g,q){function r(a){return e.contextPath()+"/rest/likes/1.0/content/"+a+"/likes"}function v(){return 0<(g.get("remote-user")||"").length&&!1!==g.get("remote-user-has-licensed-access")}function t(a,b,h){var d=g.get("remote-user")||"";b=c.Likes.LikeSummaryFactory.getLikeSummary(a.likes,b,d);b.key||!n||"page"!==a.content_type&&"blogpost"!==a.content_type||(b.text="Be the first to like this");
a=[];a.push(b.text);a=a.concat(b.args);(a=e.format.apply(e,a))&&0<a.length?(n&&h.addClass("like-summary-margin-left"),h.html(a)):(h.removeClass("like-summary-margin-left"),h.empty());a&&(h.find(".likes").click(p.showLikeUsers),c.Binder.userHover())}function m(a,b,h){var k=g.get("remote-user")||"";if(void 0===a)throw Error("type is required");if(void 0===b)throw Error("contentId is required");if(void 0===h)throw Error("contentType is required");return function(){if("object"!==typeof this||!this.nodeType||
1!==this.nodeType||"A"!==this.nodeName)throw Error("this handler should be bound to a DOM anchor element");var u=d(this),f=arguments.callee,c=u.next(".like-summary");d.ajax({type:0===a?"POST":"DELETE",url:r(b),contentType:"application/json",data:{"atlassian-token":g.get("atlassian-token")},dataType:"json",timeout:1E4}).fail(function(d){var k;405===d.status?q.showError(q.parseError(d),q.Location.FLAG):k=0===a?"Like failed":"Unlike failed";d=c.siblings(".like-error");
0===d.length&&void 0!==k?c.after('\x3cspan class\x3d"like-error" title\x3d"'+k+'"\x3e\x3c/span\x3e'):d.attr("title",k);u.off("click").on("click",null,0===a?m(0,b,h):m(1,b,h)).find(".like-button-text").html(0===a?"Like":"Unlike")}).success(function(){c.attr("data-liked",0===a);c.parent().find(".like-error").remove();t(l[b],b,c)});x(u,a,f,k,b,h,c);return!1}}function x(a,b,h,k,c,f){a.off("click",null,h).on("click",null,0===b?m(1,c,f):m(0,c,f)).find(".like-button-text").html(0===
b?"Unlike":"Like");l[c]=l[c]||{content_type:f,likes:[]};0===b?l[c].likes.push({user:{name:k}}):l[c].likes=d.grep(l[c].likes,function(a){return a.user.name!==k});0===b&&e.trigger("analytics",{name:"confluence."+f+".like.create",data:{pageID:g.get("page-id")}})}var f,l={},n=v()&&"READ_ONLY"!==g.get("access-mode"),w=d.Deferred(),p={getLikesCache:function(){return w.promise()},showLikeUsers:function(a){a&&a.preventDefault();a=d(this);var b=a.data("content-id");
a.blur();f&&(f.remove(),f=void 0);f=new e.Dialog(400,365,"likes-dialog");f.addHeader("People who like this");f.addPanel("Panel 1","\x3cdiv class\x3d'spinner-container'\x3e\x3c/div\x3e");f.addCancel("Close",function(a){a.remove();f=void 0});f.getCurrentPanel().setPadding(0);f.show();d.ajax({type:"GET",url:r(b),data:{expand:"user",max:50},dataType:"json"}).done(function(a){f.popup.element.is(":visible")&&(a.showFollowActions=v(),f.getCurrentPanel().html(c.Templates.Likes.likesDialog(a)),
d("#likes-dialog").find(".likes-dialog-follow-button").click(function(){var a=d(this);d.ajax({type:"PUT",url:e.contextPath()+"/rest/likes/1.0/user/"+g.get("remote-user")+"/following?username\x3d"+a.data("username"),contentType:"application/json",dataType:"json"}).done(function(){a.replaceWith("Following")})}))})},appendAction:function(a){a=a.find(".comment-actions-primary");var b=a.find("li[class~\x3d'comment-date']"),h=d(c.Templates.Likes.commentLikeSection({showLikeButton:n}));
0===b.length?a.find("li:last-child").after(h):b.before(h)},reload:function(a){d.each(a,function(a,b){var e=d("#comment-"+a).find(".like-summary");t(b,a,e);l[a]=b});var b=d("#page-comments");!n&&b.find(".like-summary:empty").each(function(){d(this).closest(".comment-action-like").hide()});b.find(".comment").each(function(){p.updateComment(d(this),a)});w.resolve(l)},updateComment:function(a,b){var c=a.attr("id");if(!c)return!0;var f=(/^comment-(\d+)$/.exec(c)||[])[1];if(!f)throw Error('Expecting ID attribute of comment to be in format "comment-XXX", found: '+
c);b=b[f]&&g.get("remote-user")&&0<d.grep(b[f].likes,function(a){return a.user.name===g.get("remote-user")}).length;a.find(".like-button").click(b?m(1,f,"comment"):m(0,f,"comment")).find(".like-button-text").html(b?"Unlike":"Like")},init:function(){var a=d(c.Templates.Likes.likeSection({showLikeButton:n}));g.get("page-id")&&d.ajax({type:"GET",url:r(g.get("page-id")),data:{commentLikes:!0},dataType:"json"}).done(function(b){if(g.get("remote-user")){var c=
0<d.grep(b.likes,function(a){return a.user.name===g.get("remote-user")}).length,f=b.content_type;a.find(".like-button").click(c?m(1,g.get("page-id"),f):m(0,g.get("page-id"),f)).find(".like-button-text").html(c?"Unlike":"Like")}c=a.find(".like-summary");t(b,b.content_id,c);""!==c.html()||n||a.hide();d("\x3cdiv id\x3d'likes-and-labels-container'\x3e\x3c/div\x3e").insertBefore("#labels-section").append(a).append(d("#labels-section"));l[b.content_id]=b;p.reload(b.commentLikes)});
p.appendAction(d("#page-comments"))}};return p});require("confluence/module-exporter").safeRequire("confluence-like/like",function(e){var c=require("confluence/legacy"),d=require("ajs");c.Likes.getLikesCache=e.getLikesCache;d.PageGadget||window.parent.AJS&&window.parent.AJS.PageGadget||(d.toInit(e.init),c.Likes.showLikeUsers=e.showLikeUsers,c.Likes.appendAction=e.appendAction,c.Likes.reload=e.reload,c.Likes.updateComment=e.updateComment)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-like:content-like-resources', location = '/com/atlassian/confluence/plugins/like/js/like-summary-factory.js' */
define("confluence-like/like-summary-factory",["ajs","confluence/templates","jquery"],function(a,g,h){var k={"likes.summary.you":"You like this","likes.summary.1.promoted":"{0} likes this","likes.summary.2.promoted":"{0} and {1} like this","likes.summary.3.promoted":"{0}, {1} and {2} like this","likes.summary.1.non-promoted":"{0} likes this","likes.summary.x.non-promoted":"\u003ca href=\u0022\u0022 {1}\u003e{0} people\u003c/a\u003e like this",
"likes.summary.you.1.promoted":"You and {0} like this","likes.summary.you.2.promoted":"You, {0} and {1} like this","likes.summary.you.3.promoted":"You, {0}, {1} and {2} like this","likes.summary.you.1.non-promoted":"You and {0} like this","likes.summary.you.x.non-promoted":"You and \u003ca href=\u0022\u0022 {1}\u003e{0} others\u003c/a\u003e like this","likes.summary.1.promoted.1.non-promoted":"{0} and {1} like this",
"likes.summary.1.promoted.x.non-promoted":"{0} and \u003ca href=\u0022\u0022 {2}\u003e{1} others\u003c/a\u003e like this","likes.summary.2.promoted.1.non-promoted":"{0}, {1} and {2} like this","likes.summary.2.promoted.x.non-promoted":"{0}, {1} and \u003ca href=\u0022\u0022 {3}\u003e{2} others\u003c/a\u003e like this","likes.summary.3.promoted.1.non-promoted":"{0}, {1}, {2} and {3} like this","likes.summary.3.promoted.x.non-promoted":"{0}, {1}, {2} and \u003ca href=\u0022\u0022 {4}\u003e{3} others\u003c/a\u003e like this","likes.summary.you.1.promoted.1.non-promoted":"You, {0} and {1} like this",
"likes.summary.you.1.promoted.x.non-promoted":"You, {0} and \u003ca href=\u0022\u0022 {2}\u003e{1} others\u003c/a\u003e like this","likes.summary.you.2.promoted.1.non-promoted":"You, {0}, {1} and {2} like this","likes.summary.you.2.promoted.x.non-promoted":"You, {0}, {1} and \u003ca href=\u0022\u0022 {3}\u003e{2} others\u003c/a\u003e like this","likes.summary.you.3.promoted.1.non-promoted":"You, {0}, {1}, {2} and {3} like this","likes.summary.you.3.promoted.x.non-promoted":"You, {0}, {1}, {2} and \u003ca href=\u0022\u0022 {4}\u003e{3} others\u003c/a\u003e like this"};
return{getLikeSummary:function(a,c,m){if(!a||0===a.length)return{key:"",text:""};if(!c)throw Error("contentId is required.");var l,e=[],f=[];h.each(a,function(a,b){b.user&&b.user.name==m?l=b:3>e.length&&b.user.followedByRemoteUser?e.push(b):f.push(b)});a=["likes.summary"];var d=[];null!=l&&a.push(".you");0<e.length&&(a.push("."),a.push(e.length),a.push(".promoted"),h.each(e,function(a,b){d.push(g.Likes.userLink(b))}));1===f.length?(a.push(".1.non-promoted"),d.push(g.Likes.userLink(f[0]))):1<f.length&&
(a.push(".x.non-promoted"),d.push(f.length),d.push('class\x3d"likes" data-content-id\x3d"'+c+'"'));c=a.join("");return{key:c,args:0===d.length?void 0:d,text:c in k?k[c]:""}}}});require("confluence/module-exporter").exportModuleAsGlobal("confluence-like/like-summary-factory","Confluence.Likes.LikeSummaryFactory");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-like:content-like-resources', location = '/templates/com/atlassian/confluence/plugins/like/soy/like.soy' */
// This file was automatically generated from like.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Likes.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Likes == 'undefined') { Confluence.Templates.Likes = {}; }


Confluence.Templates.Likes.likeButton = function(opt_data, opt_ignored) {
  return '<a href="" class="like-button">' + ((opt_data.useIcon) ? '<span class="aui-icon aui-icon-small aui-iconfont-like"></span>' : '') + '<span class="like-button-text">' + soy.$$escapeHtml('Like') + '</span></a>';
};
if (goog.DEBUG) {
  Confluence.Templates.Likes.likeButton.soyTemplateName = 'Confluence.Templates.Likes.likeButton';
}


Confluence.Templates.Likes.likeSection = function(opt_data, opt_ignored) {
  return '<div id="likes-section" class="no-print">' + ((opt_data.showLikeButton) ? Confluence.Templates.Likes.likeButton({useIcon: true}) : '') + '<span class="like-summary"></span></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Likes.likeSection.soyTemplateName = 'Confluence.Templates.Likes.likeSection';
}


Confluence.Templates.Likes.commentLikeSection = function(opt_data, opt_ignored) {
  return '<li class="comment-action-like">' + ((opt_data.showLikeButton) ? Confluence.Templates.Likes.likeButton({useIcon: false}) : '') + '<span class="like-summary"></span></li>';
};
if (goog.DEBUG) {
  Confluence.Templates.Likes.commentLikeSection.soyTemplateName = 'Confluence.Templates.Likes.commentLikeSection';
}


Confluence.Templates.Likes.likesDialog = function(opt_data, opt_ignored) {
  var output = '<div id="likes-dialog-body"><ol>';
  var likeList26 = opt_data.likes;
  var likeListLen26 = likeList26.length;
  for (var likeIndex26 = 0; likeIndex26 < likeListLen26; likeIndex26++) {
    var likeData26 = likeList26[likeIndex26];
    output += '<li><div class="avatar-container"><a href="' + soy.$$escapeHtml(likeData26.user.url) + '"><img class="like-user-avatar" src="' + soy.$$escapeHtml(likeData26.user.avatarUrl) + '"></a></div><div class="like-user"><a class="like-user-link" href="' + soy.$$escapeHtml(likeData26.user.url) + '">' + soy.$$escapeHtml(likeData26.user.fullName) + '</a></div>' + ((opt_data.showFollowActions) ? '<div class="follow-button-container aui-toolbar"><ul class="toolbar-group"><li class="toolbar-item">' + ((likeData26.user.followedByRemoteUser) ? soy.$$escapeHtml('Following') : '<button data-username="' + soy.$$escapeHtml(likeData26.user.name) + '" class="likes-dialog-follow-button toolbar-trigger">' + soy.$$escapeHtml('Follow') + '</button>') + '</li></ul></div>' : '') + '</li>';
  }
  output += '</ol></div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Likes.likesDialog.soyTemplateName = 'Confluence.Templates.Likes.likesDialog';
}


Confluence.Templates.Likes.userLink = function(opt_data, opt_ignored) {
  return '<a href="' + soy.$$escapeHtml(opt_data.user.url) + '" class="confluence-userlink" data-username="' + soy.$$escapeHtml(opt_data.user.name) + '">' + soy.$$escapeHtml(opt_data.user.fullName) + '</a>';
};
if (goog.DEBUG) {
  Confluence.Templates.Likes.userLink.soyTemplateName = 'Confluence.Templates.Likes.userLink';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:shared-templates', location = '/includes/soy/user.soy' */
// This file was automatically generated from user.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.User.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.User == 'undefined') { Confluence.Templates.User = {}; }


Confluence.Templates.User.userLinkUrl = function(opt_data, opt_ignored) {
  return soy.$$escapeHtml("/wiki") + '/display/~' + soy.$$escapeUri(opt_data.username);
};
if (goog.DEBUG) {
  Confluence.Templates.User.userLinkUrl.soyTemplateName = 'Confluence.Templates.User.userLinkUrl';
}


Confluence.Templates.User.logo = function(opt_data, opt_ignored) {
  return '' + ((opt_data.profilePictureInfo['default'] && opt_data.username == opt_data.currentUsername) ? '<a ' + ((opt_data.canView) ? ' ' + ((! opt_data.disableUserHover) ? 'class="userLogoLink"' : '') + ' data-username="' + soy.$$escapeHtml(opt_data.username) + '"' : '') + ' href="' + soy.$$escapeHtml("/wiki") + '/users/profile/editmyprofilepicture.action" title="' + soy.$$escapeHtml('Add a picture of yourself') + '"><img class="userLogo logo defaultLogo" src="' + soy.$$escapeHtml("/wiki/s/nb8bdl/8703/51k4y0/_") + '/images/icons/profilepics/add_profile_pic.svg" alt="' + soy.$$escapeHtml('User icon') + ': ' + soy.$$escapeHtml('Add a picture of yourself') + '"></a>' : (opt_data.profilePictureInfo.anonymous) ? '<img class="userLogo logo anonymous" src="' + soy.$$escapeHtml("/wiki/s/nb8bdl/8703/51k4y0/_") + '/images/icons/profilepics/anonymous.svg" alt="' + soy.$$escapeHtml('User icon') + ': ' + soy.$$escapeHtml('Anonymous') + '" title="' + soy.$$escapeHtml('Anonymous') + '">' : (opt_data.canView) ? '<a ' + ((! opt_data.disableUserHover) ? 'class="userLogoLink"' : '') + ' data-username="' + soy.$$escapeHtml(opt_data.username) + '" href="' + Confluence.Templates.User.userLinkUrl(opt_data) + '"><img class="userLogo logo" src="' + soy.$$escapeHtml(opt_data.profilePictureInfo.uriReference) + '" alt="' + soy.$$escapeHtml('User icon') + ': ' + soy.$$escapeHtml(opt_data.username) + '" title="' + soy.$$escapeHtml(opt_data.username) + '"></a>' : '<img class="userLogo logo anonymous" src="' + soy.$$escapeHtml("/wiki/s/nb8bdl/8703/51k4y0/_") + '/images/icons/profilepics/anonymous.svg" alt="' + soy.$$escapeHtml('User icon') + ': ' + soy.$$escapeHtml(opt_data.username) + '" title="' + soy.$$escapeHtml(opt_data.username) + '">');
};
if (goog.DEBUG) {
  Confluence.Templates.User.logo.soyTemplateName = 'Confluence.Templates.User.logo';
}


Confluence.Templates.User.usernameLink = function(opt_data, opt_ignored) {
  return '' + ((opt_data.username && opt_data.username != '') ? '<a href="' + Confluence.Templates.User.userLinkUrl(opt_data) + '"' + ((opt_data.canView) ? 'class="url fn confluence-userlink" data-username="' + soy.$$escapeHtml(opt_data.username) + '"' : 'class="url fn"') + '>' + ((opt_data.fullName && opt_data.fullName != '') ? soy.$$escapeHtml(opt_data.fullName) : soy.$$escapeHtml(opt_data.username)) + '</a>' : soy.$$escapeHtml('Anonymous'));
};
if (goog.DEBUG) {
  Confluence.Templates.User.usernameLink.soyTemplateName = 'Confluence.Templates.User.usernameLink';
}


Confluence.Templates.User.fullNameOrAnonymous = function(opt_data, opt_ignored) {
  return '' + ((opt_data.user && opt_data.user.fullName) ? soy.$$escapeHtml(opt_data.user.fullName) : soy.$$escapeHtml('Anonymous'));
};
if (goog.DEBUG) {
  Confluence.Templates.User.fullNameOrAnonymous.soyTemplateName = 'Confluence.Templates.User.fullNameOrAnonymous';
}


Confluence.Templates.User.usernameOrAnonymous = function(opt_data, opt_ignored) {
  return '' + ((opt_data.user && opt_data.user.name) ? soy.$$escapeHtml(opt_data.user.name) : soy.$$escapeHtml('Anonymous'));
};
if (goog.DEBUG) {
  Confluence.Templates.User.usernameOrAnonymous.soyTemplateName = 'Confluence.Templates.User.usernameOrAnonymous';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:shared-templates', location = '/includes/soy/icons.soy' */
// This file was automatically generated from icons.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Icons.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Icons == 'undefined') { Confluence.Templates.Icons = {}; }


Confluence.Templates.Icons.contentIcon = function(opt_data, opt_ignored) {
  return '<a' + ((opt_data.linkColour) ? ' ' + soy.$$escapeHtml(opt_data.linkColour) : '') + ' href="' + soy.$$escapeHtml(opt_data.linkHref) + '" class="icon ' + soy.$$escapeHtml(opt_data.iconCss) + '" title="' + soy.$$escapeHtml(opt_data.iconTitle) + '">' + soy.$$escapeHtml(opt_data.iconTitle) + '</a>';
};
if (goog.DEBUG) {
  Confluence.Templates.Icons.contentIcon.soyTemplateName = 'Confluence.Templates.Icons.contentIcon';
}


Confluence.Templates.Icons.contentIconFont = function(opt_data, opt_ignored) {
  return '<a' + ((opt_data.linkColour) ? ' ' + soy.$$escapeHtml(opt_data.linkColour) : '') + ' href="' + soy.$$escapeHtml(opt_data.linkHref) + '" title="' + soy.$$escapeHtml(opt_data.iconTitle) + '"><span class="aui-icon ' + soy.$$escapeHtml(opt_data.iconCss) + '">' + soy.$$escapeHtml(opt_data.iconTitle) + '</span></a>';
};
if (goog.DEBUG) {
  Confluence.Templates.Icons.contentIconFont.soyTemplateName = 'Confluence.Templates.Icons.contentIconFont';
}


Confluence.Templates.Icons.iconSpan = function(opt_data, opt_ignored) {
  opt_data = opt_data || {};
  return '<span class="icon' + ((opt_data.type) ? ' aui-icon aui-icon-small aui-iconfont-' + soy.$$escapeHtml(opt_data.type) : '') + ' ' + ((opt_data.additionalClasses) ? soy.$$escapeHtml(opt_data.additionalClasses) : '') + '">' + ((opt_data.text) ? soy.$$escapeHtml(opt_data.text) : '') + '</span>';
};
if (goog.DEBUG) {
  Confluence.Templates.Icons.iconSpan.soyTemplateName = 'Confluence.Templates.Icons.iconSpan';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:shared-templates', location = '/includes/soy/captcha.soy' */
// This file was automatically generated from captcha.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Captcha.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Captcha == 'undefined') { Confluence.Templates.Captcha = {}; }


Confluence.Templates.Captcha.form = function(opt_data, opt_ignored) {
  var output = '<div class="captcha field-group"><label id="captcha-response-label" for="captcha-response"><span class="assistive">' + soy.$$escapeHtml('If you are unable to use this CAPTCHA please \x3ca href\x3d\x22administrators.action\x22 tabindex\x3d\x225\x22\x3econtact your administrator\x3c/a\x3e for assistance.') + '</span></label>' + Confluence.Templates.Captcha.image(opt_data) + '<input type="text" id="captcha-response" name="captchaResponse" value="" class="text" placeholder="' + soy.$$escapeHtml('Type the word above') + '">';
  if (opt_data.captchaErrors && opt_data.captchaErrors.length) {
    var errorList13 = opt_data.captchaErrors;
    var errorListLen13 = errorList13.length;
    for (var errorIndex13 = 0; errorIndex13 < errorListLen13; errorIndex13++) {
      var errorData13 = errorList13[errorIndex13];
      output += aui.message.warning({content: errorData13});
    }
  }
  output += '</div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Captcha.form.soyTemplateName = 'Confluence.Templates.Captcha.form';
}


Confluence.Templates.Captcha.image = function(opt_data, opt_ignored) {
  return '<img src="' + soy.$$escapeHtml("/wiki") + '/jcaptcha?id=' + soy.$$escapeHtml(opt_data.captchaId) + '" class="captcha-image" alt="' + soy.$$escapeHtml('CAPTCHA image') + '"><input type="hidden" name="captchaId" value="' + soy.$$escapeHtml(opt_data.captchaId) + '" placeholder="' + soy.$$escapeHtml('Type the word above') + '">';
};
if (goog.DEBUG) {
  Confluence.Templates.Captcha.image.soyTemplateName = 'Confluence.Templates.Captcha.image';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:shared-templates', location = '/includes/soy/notifications.soy' */
// This file was automatically generated from notifications.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Notifications.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Notifications == 'undefined') { Confluence.Templates.Notifications = {}; }


Confluence.Templates.Notifications.notLoggedIn = function(opt_data, opt_ignored) {
  return '' + ((! opt_data.isUserAuthenticated) ? '<div id="anonymous-warning" class="aui-message aui-message-warning closeable">' + soy.$$filterNoAutoescape('You are not logged in. Any changes you make will be marked as \x3cspan class\x3d\x22smalltext\x22\x3eanonymous\x3c/span\x3e.') + ((! opt_data.isExternalUserManagementEnabled) ? ' ' + soy.$$filterNoAutoescape(AJS.format('You may want to \x3ca href\x3d\x22{0}\x22\x3eLog In\x3c/a\x3e if you already have an account.',opt_data.loginURL)) : '') + '</div>' : '');
};
if (goog.DEBUG) {
  Confluence.Templates.Notifications.notLoggedIn.soyTemplateName = 'Confluence.Templates.Notifications.notLoggedIn';
}


Confluence.Templates.Notifications.actionErrors = function(opt_data, opt_ignored) {
  var output = '';
  if (opt_data.actionErrors.length > 0) {
    output += '<div class="aui-message aui-message-error ' + ((opt_data.closeable) ? 'closeable' : '') + '"><p class="title">' + soy.$$escapeHtml('The following error(s) occurred:') + '</p><ul>';
    var errorHtmlList24 = opt_data.actionErrors;
    var errorHtmlListLen24 = errorHtmlList24.length;
    for (var errorHtmlIndex24 = 0; errorHtmlIndex24 < errorHtmlListLen24; errorHtmlIndex24++) {
      var errorHtmlData24 = errorHtmlList24[errorHtmlIndex24];
      output += '<li>' + soy.$$filterNoAutoescape(errorHtmlData24) + '</li>';
    }
    output += '</ul></div>';
  }
  output += '<div id="action-messages-notifications"></div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Notifications.actionErrors.soyTemplateName = 'Confluence.Templates.Notifications.actionErrors';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.browser.metrics.browser-metrics-plugin:api', location = 'probe.js' */
!function(){var n,t,e,r,i,o,u,a,c,s,f,l,h,d,y,p,v,m,w;n=function(){return window}(),t=function(n){return!(!n.performance||!n.performance.now)}(n),e=[],r=function(n){return function(t){n.unshift({addReporter:t})}}(e),i=function(n){return function(t){for(;n.length;)t(n.splice(0,1)[0]);n.unshift=t,n.push=t}}(e),o=function(n,t){return function(e){n.push({end:{key:e.key,timestamp:t.performance.now()}})}}(e,n),u=function(n){return n.document}(n),a=function(n){return n.Promise}(n),c=function(){function n(){this._={}}var t=function(e){var r=e[0],i=e[1];i instanceof n?e.length>=3?Object.keys(i._).forEach(function(n){t([r,i._[n],n].concat(e.slice(2)))}):Object.keys(i._).forEach(function(n){t([r,i._[n],n])}):Array.isArray(i)&&r.apply(null,[i].concat(e.slice(2)))};n.prototype.forEach=function(n){t([n,this])},n.prototype.add=function(){for(var t=this,e=null,r=null,i=0;i<arguments.length;i++){if(r=arguments[i],i===arguments.length-1&&Array.isArray(t)){t.push(r);break}i<arguments.length-2&&!t._.hasOwnProperty(r)?t._[r]=new n:i!==arguments.length-2||t._.hasOwnProperty(r)||(t._[r]=[]),t=t._[r],e=r}};var e=function(n,t){if(0!==n.length){var r=n.pop(),i=r[0],o=r[1];i===t?e(n,i):o._.hasOwnProperty(t)&&delete o._[t],0===Object.keys(o).length&&e(n,i)}};return n.prototype.remove=function(){for(var n,t=!1,r=null,i=this,o=[[r,i]],u=null,a=0;a<arguments.length;a++)if(u=arguments[a],Array.isArray(i))n=i.indexOf(u),n>-1&&(i.splice(n,1),0===i.length&&o.length>1&&e(o,r),t=!0);else{if(!i._.hasOwnProperty(u))break;a===arguments.length-1&&(delete i._[u],0===Object.keys(i).length&&o.length>1&&e(o,r),t=!0),r=u,i=i._[u],o.push([r,i])}return t},n.prototype.get=function(n){return this._.hasOwnProperty(n)?this._[n]:[]},n}(),s=function(n,t,e,r){function i(n){return!n||null==n||"null"===n||"undefined"===n}function o(t,e,r){l||(c.observe(n,{attributes:!0,childList:!0,subtree:!0}),l=!0),s.add(t,e,r)}function u(t,e){var r=n.querySelectorAll(t);return r.length&&(i(e)||Array.prototype.every.call(r,function(n){return!n.querySelector(e)}))}function a(n,e){var r;n.forEach||(n=[n]),!i(e)&&Array.isArray(e)&&(e=e.join(", "));var a=new t(function(i,a){var c=[],f=[];n.forEach(function(n){var r,i;u(n,e)||(r=new t(function(t){o(n,e,t),i=function(){s.remove(n,e,t)},f.push(i)}),c.push(r))});var l=function(){f.forEach(function(n){n()})};t.all(c).then(l).then(i,a),r=function(){l(),a()}});return a.dismiss=r,a}var c,s,f=r.MutationObserver,l=!1;return f&&t?(s=new e,c=new f(function(){s.forEach(function(n,t,e){u(e,t)&&(n.forEach(function(n){n()}),s.remove(e,t))})}),a):void 0}(u,a,c,n),f=function(n){return!!n}(s),l=function(n){function t(){c(),n.body.classList.add(u)}function e(){function e(){n.body.classList.remove(u),n.removeEventListener(i,s),n.removeEventListener(o,c),r=null}if(r)return r;var c,s,f=!1;return r=new Promise(function(e,r){"visible"!==n.visibilityState?r():(s=function(){f=!0},c=function(n){n.animationName===a&&(f?r():e())},n.addEventListener(i,s),n.addEventListener(o,c),t())}),r.then(e,e),r}var r,i="visibilitychange",o="animationend",u="browser-metrics-visibility-test",a="browser-metrics-visibility-animation",c=function(){var t=n.createElement("style"),e=["."+u+" {","-webkit-animation-duration: 0.001s;","animation-duration: 0.001s;","-webkit-animation-name: "+a+";","animation-name: "+a+";","-webkit-animation-iteration-count: 1;","animation-iteration-count: 1;","}","@keyframes "+a+" {}","@-webkit-keyframes "+a+" {","from {}","to {}","}"].join("\n");t.type="text/css",t.styleSheet?t.styleSheet.cssText=e:t.appendChild(n.createTextNode(e)),n.head.appendChild(t),c=function(){}};return e}(u),h=function(n,t,e,r,i){function o(n){return Array.isArray(n)||(n=[n]),n.map(function(n){return"string"==typeof n?{selector:n,hasNone:null}:n})}function u(n){return Array.isArray(n)||"string"==typeof n}function a(n){return u(n)&&(n={rules:n}),n.rules=o(n.rules),n.requirePaint="undefined"==typeof n.requirePaint?!1:n.requirePaint,n}return function(i,o){if(n){i=a(i);var u=function(){},c=new e(function(n,r){var o=[],a=i.rules.map(function(n){var e=new t(n.selector,n.hasNone);return o.push(function(){e.dismiss()}),e});u=function(){o.forEach(function(n){n()}),r()},e.all(a).then(function(n){}).then(n,r)});return c.cancel=u,i.requirePaint&&(c=c.then(r)),"function"==typeof o&&c.then(o),c}}}(f,s,a,l,n),d=function(n,t){function e(){return r}var r=!1;return n.addEventListener("DOMContentLoaded",function(){t.setTimeout(function(){r=!0})}),e}(u,n),y=function(n,t,e,r,i,o,u){function a(){c=null}var c;return function(o){var s="isInitial"in o?o.isInitial:i()===!1,f="threshold"in o?o.threshold:1e3,l="reporters"in o?o.reporters:[];r.push({start:{key:o.key,isInitial:s,threshold:f,timestamp:s?0:u.performance.now(),reporters:Array.isArray(l)?l:[l]}}),c&&(c.cancel(),a()),o.ready&&e&&(c=n(o.ready),c.then(function(){t({key:o.key})}).then(a,a))}}(h,o,f,e,d,a,n),p=function(n){return function(t){n.push({subscribe:t})}}(e),v=function(){return window}(),m=function(n){return n.performance}(v),w=function(n,t,e,r,i,o,u){var a=function(){};return u?{start:n?i:a,end:n?r:a,addReporter:n?t:a,delegateTo:n?e:a,subscribe:n?o:a}:void 0}(t,r,i,o,y,p,m),window["browser-metrics"]=w,window.define&&window.define("internal/browser-metrics",function(){return w})}();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.browser.metrics.browser-metrics-plugin:api', location = 'internal/browser-metrics-aa-beacon.js' */
!function(){var e={};e=function(e){function n(e,n){Object.keys(n).forEach(function(r){e[r]=n[r]})}Object.defineProperty(e,"__esModule",{value:!0});var r,t=[],o=[];return e.addUrlCleaner=function(e){o.push(e)},e.cleanUrl=function(e){return o.reduce(function(n,r){var t=r(e);return t.length>n.length?t:n},"")},e.addReportMarshaller=function(e){t.push(e)},e.setEventQueue=function(e){r=e},e.beacon=function(e){var o={};t.forEach(function(r){var t=r(e);"object"==typeof t&&n(o,t)});var a={name:"browser.metrics.navigation",properties:o};(r||AJS.EventQueue).push(a)},e}(e),window["browser-metrics-aa-beacon"]=e,window.define&&window.define("internal/browser-metrics-aa-beacon",function(){return e})}();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.browser.metrics.browser-metrics-plugin:api', location = 'loader.js' */
!function(){var n={},r={};r=function(n,r,i){function e(){t===u&&o&&(o(),o=null)}Object.defineProperty(n,"__esModule",{value:!0});var t=0,u=0,o=null,c={install:function(n){t+=1,n(function(){u+=1,e()})}};return r["browser-metrics-plugin"]=c,i.require(["wrc!browser-metrics-plugin.contrib"],function(){r.require(["internal/browser-metrics-plugin/collector"],function(n){o=function(){n.install()},e()})}),n}(r,n=window,n.WRM)}();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-browser-metrics:viewcontent', location = '/js/viewcontent.js' */
require(["confluence/api/event","internal/browser-metrics","ajs","jquery"],function(f,e,c,g){function b(i){if(!g(i.target).hasClass("full-load")){var h=i.data.type;if(c.Meta.getBoolean("collaborative-content")){var j="confluence."+h+".edit.collaborative.quick-view";e.start({key:j});c.bind("rte-collab-ready",function(){e.end({key:j});e.start({key:j+".connected"})});c.bind("synchrony.connected",function(){e.end({key:j+".connected"})})}else{if(!c.Meta.getBoolean("shared-drafts")){e.start({key:"confluence."+h+".edit.quick-view",ready:".active-richtext"})}}}}function a(){e.start({key:"confluence.file.preview.firstpage"});f.bind("confluence-previews.fileviewer.completed",function(){e.end({key:"confluence.file.preview.firstpage"})})}function d(){var j=c.Meta.get("content-type");var i=g("#confluence-ui.confluence-dashboard").length;var h=window.location.href.indexOf("/content-only/")>-1;if(j){if(!i&&!h){e.start({key:"confluence."+j+".view",ready:".wiki-content",isInitial:true})}g("#editPageLink").on("click",{type:j},b);g(".confluence-embedded-file-wrapper").on("click",a)}}c.toInit(d)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.gatekeeper.gatekeeper-plugin:who-can-view-entry', location = 'who-can-view-entry.js' */
!function(e){function n(n){for(var t,r,a=n[0],u=n[1],i=0,l=[];i<a.length;i++)r=a[i],Object.prototype.hasOwnProperty.call(o,r)&&o[r]&&l.push(o[r][0]),o[r]=0;for(t in u)Object.prototype.hasOwnProperty.call(u,t)&&(e[t]=u[t]);for(c&&c(n);l.length;)l.shift()()}var t={},o={"who-can-view-entry":0};function r(n){if(t[n])return t[n].exports;var o=t[n]={i:n,l:!1,exports:{}};return e[n].call(o.exports,o,o.exports,r),o.l=!0,o.exports}r.e=function(e){var n=[];return 0===o[e]?Promise.resolve():o[e]?o[e][2]:(n.push(new Promise((function(n,t){o[e]=[n,t]})),new Promise((function(n,t){WRM.require("wrc!com.atlassian.confluence.plugins.gatekeeper.gatekeeper-plugin:"+e).then(n,t)}))),o[e][2]=Promise.all(n))},r.m=e,r.c=t,r.d=function(e,n,t){r.o(e,n)||Object.defineProperty(e,n,{enumerable:!0,get:t})},r.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},r.t=function(e,n){if(1&n&&(e=r(e)),8&n)return e;if(4&n&&"object"==typeof e&&e&&e.__esModule)return e;var t=Object.create(null);if(r.r(t),Object.defineProperty(t,"default",{enumerable:!0,value:e}),2&n&&"string"!=typeof e)for(var o in e)r.d(t,o,function(n){return e[n]}.bind(null,o));return t},r.n=function(e){var n=e&&e.__esModule?function(){return e.default}:function(){return e};return r.d(n,"a",n),n},r.o=function(e,n){return Object.prototype.hasOwnProperty.call(e,n)},r.p="",r.oe=function(e){throw console.error(e),e},"undefined"!=typeof AJS&&(r.p=AJS.contextPath()+"/download/resources/com.atlassian.confluence.plugins.gatekeeper.gatekeeper-plugin:assets-cf4b4b2a-b2cb-4d74-baac-9fefbb77d88f/");var a=window.atlassianWebpackJsonp714631d0c4f9ce0999d5f515f37461f0=window.atlassianWebpackJsonp714631d0c4f9ce0999d5f515f37461f0||[],u=a.push.bind(a);a.push=n,a=a.slice();for(var i=0;i<a.length;i++)n(a[i]);var c=u;r(r.s=418)}({174:function(e,n){e.exports=require("ajs")},418:function(e,n,t){e.exports=t(419)},419:function(e,n,t){"use strict";var o,r=(o=t(174))&&o.__esModule?o:{default:o};"undefined"!=typeof reactHotLoaderGlobal&&reactHotLoaderGlobal.default.signature;r.default.toInit((function(){var e=document.getElementById("who-can-view-button-ak-button");null!==e&&e.addEventListener("click",(function(e){e.preventDefault(),Promise.all([t.e("vendors~who-can-view-controller"),t.e("who-can-view-controller")]).then(t.t.bind(null,421,7)).then((function(e){(0,e.show)(!0)}))}))}))}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.ext.newcode-macro-plugin:syntaxhighlighter-init', location = 'sh/asyncLoader.js' */
/**
 * Load the Code Macro resources via WRM.require(). This avoids any extra blocking resources being added to the <head>
 */
define('confluence/code-macro/async-loader', ['jquery', 'wrm', 'underscore'], function ($, WRM, _) {
    return function () {
        _.defer(function () {
            const $controlButtons = $('.codeHeader .collapse-source');
            _showSpinner($controlButtons);
            const $codeBlocks = $('#content, div.cq-viewquestion, div.view-template').find('pre.syntaxhighlighter-pre');

            const wrmResources = [];
            const wrmPrefix = 'wr!com.atlassian.confluence.ext.newcode-macro-plugin:';

            if ($codeBlocks.length > 0) {
                const themeResourcePrefix = wrmPrefix + 'sh-theme-';
                wrmResources.push('wrc!code-macro');

                $codeBlocks.each(function () {
                    wrmResources.push(themeResourcePrefix + $(this).data('theme').toLowerCase());

                    // If the macro is using a custom language, we need to download it
                    const customLanguageResource = $(this).data('custom-language-resource');
                    if (typeof customLanguageResource !== 'undefined' && wrmResources.indexOf(customLanguageResource) === -1) {
                        wrmResources.push('wr!' + customLanguageResource);
                    }
                });
            } else {
                // Only the bidi marker resource
                wrmResources.push(wrmPrefix + 'code-macro-bidi');
            }

            WRM.require(wrmResources).done(function () {
                if (window.SyntaxHighlighter && typeof window.SyntaxHighlighter.highlight === 'function') {
                    window.SyntaxHighlighter.highlight();
                }
                _hideSpinner($controlButtons);
            });
        });

        /**
         * Hide the expand/collapse buttons until the highlighter is finished, and show a spinner instead
         * @param $controlButtons The expand/collapse buttons
         * @private
         */
        function _showSpinner($controlButtons) {
            $controlButtons.hide();
            // Avoiding searching the entire dom again
            _.forEach($controlButtons, function (button) {
                $(button).next('.collapse-spinner-wrapper').spin();
            });
        }

        /**
         * Show the expand/collapse buttons and hide the spinner once the highlighter is ready.
         * @param $controlButtons The expand/collapse buttons
         * @private
         */
        function _hideSpinner($controlButtons) {
            $controlButtons.show();
            $controlButtons.next('.collapse-spinner-wrapper').remove();
        }
    };
});

require('confluence/module-exporter').safeRequire('confluence/code-macro/async-loader', function (CodeMacroLoader) {
    AJS.toInit(CodeMacroLoader);
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugin.jslibs:skate-0.12.6', location = 'libs/skate/0.12.6/skate-0.12.6.js' */
(function e(g,c,j){function h(a,k){if(!c[a]){if(!g[a]){var d="function"==typeof require&&require;if(!k&&d)return d(a,!0);if(f)return f(a,!0);d=Error("Cannot find module '"+a+"'");throw d.code="MODULE_NOT_FOUND",d;}d=c[a]={exports:{}};g[a][0].call(d.exports,function(c){var f=g[a][1][c];return h(f?f:c)},d,d.exports,e,g,c,j)}return c[a].exports}for(var f="function"==typeof require&&require,d=0;d<j.length;d++)h(j[d]);return h})({1:[function(b,g,c){Object.defineProperty(c,"__esModule",{value:!0});"use strict";
c.ATTR_IGNORE="data-skate-ignore";c.types={ANY:"act",ATTR:"a",CLASS:"c",NOATTR:"ct",NOCLASS:"at",NOTAG:"ac",TAG:"t"}},{}],2:[function(b,g){g.exports={get:function(c,b){if(c.__SKATE_DATA)return c.__SKATE_DATA[b]},set:function(c,b,h){c.__SKATE_DATA||(c.__SKATE_DATA={});c.__SKATE_DATA[b]=h;return c}}},{}],3:[function(b,g){function c(c){for(var f=c.length,b=0;b<f;b++){var i=c[b],o=i.addedNodes,i=i.removedNodes;o&&(o.length&&!p(o[0].parentNode))&&d(o);i&&i.length&&a(i)}}var j=function(c){return c&&c.__esModule?
c["default"]:c},h=j(b("./globals")),f=b("./lifecycle"),d=f.initElements,a=f.removeElements,k=j(b("./mutation-observer")),p=b("./utils").getClosestIgnoredElement;g.exports={register:function(a){a&&(k.fixIe(),this.unregister());h.observer||(a=new k(c),a.observe(document,{childList:!0,subtree:!0}),h.observer=a);return this},unregister:function(){h.observer&&(h.observer.disconnect(),h.observer=void 0);return this}}},{"./globals":4,"./lifecycle":5,"./mutation-observer":6,"./utils":9}],4:[function(b,g){window.__skate||
(window.__skate={observer:void 0,registry:{}});g.exports=window.__skate},{}],5:[function(b,g,c){function j(o,c,a){if(k.get(o,c.id+":lifecycle:"+a))return!0;k.set(o,c.id+":lifecycle:"+a,!0);return!1}function h(c,a){if(!j(c,a,"created")){m(c,a.prototype,!0);a.template&&a.template(c);var b=function(a,b){return function(f){if(!b)return a(c,f,c);for(var d=f.target;d&&d!==document&&d!==c.parentNode;){if(i.call(d,b))return a(c,f,d);d=d.parentNode}}};"object"===typeof a.events&&l(a.events,function(a,d){var f,
i;i=d.split(" ");f=i.shift();i=i.join(" ");c.addEventListener(f,b(a,i),!!i&&("blur"===f||"focus"===f))});var f=function(i,f,d,b){var k;a.attributes&&a.attributes[f]&&"function"===typeof a.attributes[f][i]?k=a.attributes[f][i]:a.attributes&&"function"===typeof a.attributes[f]?k=a.attributes[f]:"function"===typeof a.attributes&&(k=a.attributes);k&&k(c,{type:i,name:f,newValue:d,oldValue:b})},d,k=c.attributes,h=[],g=k.length;(new p(function(a){a.forEach(function(a){var c,i=a.attributeName,d=k[i];d&&null===
a.oldValue?c="created":d&&null!==a.oldValue?c="updated":d||(c="removed");f(c,i,d?d.value||d.nodeValue:void 0,a.oldValue)})})).observe(c,{attributes:!0,attributeOldValue:!0});for(d=0;d<g;d++)h.push(k[d]);for(d=0;d<g;d++){var n=h[d];f("created",n.nodeName,n.value||n.nodeValue)}a.created&&a.created(c)}}function f(c){for(var i=c.length,d=0;d<i;d++){var b=c[d];if(!(1!==b.nodeType||b.attributes[a])){for(var k=n.getForElement(b),g=k.length,m=0;m<g;m++){var l=b,p=k[m];h(l,p);j(l,p,"attached")||(l.removeAttribute(p.unresolvedAttribute),
l.setAttribute(p.resolvedAttribute,""),p.attached&&p.attached(l))}b=b.childNodes;b.length&&f(b)}}}function d(a){for(var c=a.length,i=0;i<c;i++){var f=a[i];if(1===f.nodeType){d(f.childNodes);for(var b=n.getForElement(f),g=b.length,j=0;j<g;j++){var h=f,m=b[j];m.detached&&m.detached(h);k.set(h,m.id+":lifecycle:attached",!1)}}}}g=function(a){return a&&a.__esModule?a["default"]:a};Object.defineProperty(c,"__esModule",{value:!0});"use strict";var a=b("./constants").ATTR_IGNORE,k=g(b("./data")),p=g(b("./mutation-observer")),
n=g(b("./registry")),b=b("./utils"),m=b.inherit,l=b.objEach,b=window.HTMLElement.prototype,i=b.matches||b.msMatchesSelector||b.webkitMatchesSelector||b.mozMatchesSelector||b.oMatchesSelector;c.triggerCreated=h;c.initElements=f;c.removeElements=d},{"./constants":1,"./data":2,"./mutation-observer":6,"./registry":7,"./utils":9}],6:[function(b,g){function c(a,c){return{addedNodes:null,attributeName:null,attributeNamespace:null,nextSibling:null,oldValue:null,previousSibling:null,removedNodes:null,target:a,
type:c||"childList"}}function j(a,c){var f=a.childNodes;if(f)for(var d=f.length,b=0;b<d;b++){var k=f[b];c(k);j(k,c)}}function h(a){if(n&&!m)return new n(a);this.callback=a;this.elements=[]}var f=b("./utils"),d=f.debounce,a=f.objEach,k=window.HTMLElement.prototype,p=window.HTMLElement.prototype.contains,n=window.MutationObserver||window.WebkitMutationObserver||window.MozMutationObserver,m=!1,l=-1<window.navigator.userAgent.indexOf("Trident");h.fixIe=function(){if(l&&!m){var a=Object.getOwnPropertyDescriptor(k,
"innerHTML");Object.defineProperty(k,"innerHTML",{get:function(){return a.get.call(this)},set:function(c){j(this,function(a){var c=document.createEvent("MutationEvent");c.initMutationEvent("DOMNodeRemoved",!0,!1,null,null,null,null,null);a.dispatchEvent(c)});a.set.call(this,c)}});m=!0}};Object.defineProperty(h,"isFixingIe",{get:function(){return m}});h.prototype={observe:function(a,f){function b(a){h.push(a);n()}var k=this,j,g,h=[],l=[],n=d(function(){for(var b=h.length,d=0;d<b;d++){var n=h[d],r=
n.target;if(r){var n=n.type,q=r.parentNode;if(f.childList&&(f.subtree||q===a)){var s=m&&"DOMNodeRemoved"===n,t=j&&(1!==j.nodeType?!1:j.contains?j.contains(r):p.call(j,r));if(s||!t){if(!g||g.target!==q)l.push(g=c(q));"DOMNodeInserted"===n?(g.addedNodes||(g.addedNodes=[]),g.addedNodes.push(r)):(g.removedNodes||(g.removedNodes=[]),g.removedNodes.push(r));j=r}}}}k.callback(l);h=[];l=[];g=j=void 0}),t={},q=[],u=d(function(){var a=q.length;k.callback(q);q.splice(0,a)}),s={target:a,options:f,insertHandler:b,
removeHandler:b,attributeHandler:function(d){var b=d.target;if(f.attributes&&(f.subtree||b===a)){var k=d.attrName,g=d.prevValue,d=d.newValue,b=c(b,"attributes");b.attributeName=k;f.attributeOldValue&&(b.oldValue=t[k]||g||null);q.push(b);f.attributeOldValue&&(t[k]=d);u()}}};this.elements.push(s);f.childList&&(a.addEventListener("DOMNodeInserted",s.insertHandler),a.addEventListener("DOMNodeRemoved",s.removeHandler));f.attributes&&a.addEventListener("DOMAttrModified",s.attributeHandler);return this},
disconnect:function(){a(this.elements,function(a){a.target.removeEventListener("DOMNodeInserted",a.insertHandler);a.target.removeEventListener("DOMNodeRemoved",a.removeHandler);a.target.removeEventListener("DOMAttrModified",a.attributeHandler)});this.elements=[];return this}};g.exports=h},{"./utils":9}],7:[function(b,g){function c(a,c){return d(j.registry,a)&&-1<j.registry[a].type.indexOf(c)}var j,h=b("./globals");j=h&&h.__esModule?h["default"]:h;var f=b("./constants.js").types,d=b("./utils").hasOwn;
g.exports={clear:function(){j.registry={};return this},getForElement:function(a){var d=a.attributes,b=d.length,g=[],h=d.is,l=h&&(h.value||h.nodeValue),h=a.tagName.toLowerCase(),i=l||h,o;c(i,f.TAG)&&(i=j.registry[i],o=i["extends"],l?h===o&&g.push(i):o||g.push(i));for(l=0;l<b;l++)i=d[l].nodeName,c(i,f.ATTR)&&(i=j.registry[i],o=i["extends"],(!o||h===o)&&g.push(i));(d=a.classList)?a=d:(a=a.attributes,a=a["class"]&&a["class"].nodeValue.split(/\s+/)||[]);d=a.length;for(b=0;b<d;b++)i=a[b],c(i,f.CLASS)&&
(i=j.registry[i],o=i["extends"],(!o||h===o)&&g.push(i));return g},has:function(a){return d(j.registry,a)},set:function(a,c){if(this.has(a))throw Error('A definition of type "'+c.type+'" with the ID of "'+a+'" already exists.');j.registry[a]=c;return this},remove:function(a){this.has(a)&&delete j.registry[a];return this}}},{"./constants.js":1,"./globals":4,"./utils":9}],8:[function(b,g){function c(a,b){b=n(b||{},c.defaults);b.id=a;if(k.has(b.id))throw Error('A definition of type "'+b.type+'" with the ID of "'+
a+'" already exists.');k.set(b.id,b);i();h.register(b.remove);if(-1<b.type.indexOf(m.TAG)){var f=b,g=function(){var a;a=f["extends"];var b=f.id;a?(a=document.createElement(a),a.setAttribute("is",b)):a=document.createElement(b);f.prototype=g.prototype;d(a,f);return a};g.prototype=f.prototype;return g}}var j=function(a){return a&&a.__esModule?a["default"]:a},h=j(b("./document-observer")),f=b("./lifecycle"),d=f.triggerCreated,a=f.initElements,k=j(b("./registry")),p=b("./utils"),f=p.debounce,n=p.inherit,
j=j(b("./version")),m=b("./constants").types,l=!!document.attachEvent,i=f(function(){var b=function(){a(document.getElementsByTagName("html"))},c;c=l?"complete"===document.readyState:"interactive"===document.readyState||"complete"===document.readyState;c?b():l?window.addEventListener("load",b):document.addEventListener("DOMContentLoaded",b)});c.init=function(b){if(b)return"string"===typeof b&&(b=document.querySelectorAll(b)),a("undefined"===typeof b.length?[b]:b),b};c.types=m;c.version=j;c.defaults=
{attributes:void 0,events:void 0,"extends":"",id:"",prototype:{},resolvedAttribute:"resolved",template:void 0,type:m.ANY,unresolvedAttribute:"unresolved"};var o=window.skate;c.noConflict=function(){window.skate=o;return this};window.skate=c;g.exports=c},{"./constants":1,"./document-observer":3,"./lifecycle":5,"./registry":7,"./utils":9,"./version":10}],9:[function(b,g,c){function j(b,c){return Object.prototype.hasOwnProperty.call(b,c)}c.hasOwn=j;c.debounce=function(b){var c=!1;return function(){c||
(c=!0,setTimeout(function(){c=!1;b()},1))}};c.getClosestIgnoredElement=function(b){for(;b&&b!==document&&!(b instanceof DocumentFragment);){if(b.hasAttribute(h))return b;b=b.parentNode}};c.inherit=function(b,c,a){for(var g=Object.getOwnPropertyNames(c),h=g.length,j=0;j<h;j++){var m=g[j];if(a||void 0===b[m]){var l=Object.getOwnPropertyDescriptor(c,m);l.get||l.set||!l.writable||!l.enumerable||!l.configurable?Object.defineProperty(b,m,l):b[m]=c[m]}}return b};c.objEach=function(b,c){for(var a in b)j(b,
a)&&c(b[a],a)};Object.defineProperty(c,"__esModule",{value:!0});"use strict";var h=b("./constants").ATTR_IGNORE},{"./constants":1}],10:[function(b,g){g.exports="0.12.6"},{}]},{},[8]);(function(b){var g=b.skate.noConflict();define("atlassian/libs/skate-0.12.6",[],function(){return g})})(window);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.aui.staging:skate', location = '/includes/js/amd/shim/skate-amd.js' */
define("skate",["atlassian/libs/skate-0.12.6"],function(a){return a});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-page-banner:soy-resources', location = 'soy/page-banner.soy' */
// This file was automatically generated from page-banner.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.PageBanner.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.PageBanner == 'undefined') { Confluence.Templates.PageBanner = {}; }


Confluence.Templates.PageBanner.banner = function(opt_data, opt_ignored) {
  var output = '<div id="page-metadata-banner"><ul class="banner">' + Confluence.Templates.PageBanner.renderSystemContentItems(opt_data);
  var itemList6 = opt_data.pageBannerItems;
  var itemListLen6 = itemList6.length;
  for (var itemIndex6 = 0; itemIndex6 < itemListLen6; itemIndex6++) {
    var itemData6 = itemList6[itemIndex6];
    output += Confluence.Templates.PageBanner.renderPageBannerItem(itemData6);
  }
  output += '</ul></div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.PageBanner.banner.soyTemplateName = 'Confluence.Templates.PageBanner.banner';
}


Confluence.Templates.PageBanner.renderSystemContentItems = function(opt_data, opt_ignored) {
  var output = '<li id="system-content-items" class="noprint">';
  var itemList12 = opt_data.systemContentItems;
  var itemListLen12 = itemList12.length;
  for (var itemIndex12 = 0; itemIndex12 < itemListLen12; itemIndex12++) {
    var itemData12 = itemList12[itemIndex12];
    output += Confluence.Templates.PageBanner.itemAnchor(soy.$$augmentMap(itemData12, {isSystemContentItem: true}));
  }
  output += '</li>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.PageBanner.renderSystemContentItems.soyTemplateName = 'Confluence.Templates.PageBanner.renderSystemContentItems';
}


Confluence.Templates.PageBanner.renderPageBannerItem = function(opt_data, opt_ignored) {
  return '<li class="' + ((! opt_data.suppressDefaultStyle) ? 'page-metadata-item noprint' : '') + ((opt_data.isAuiButton) ? 'has-button' : '') + '" ' + ((opt_data.linkId) ? ' id="' + soy.$$escapeHtml(opt_data.linkId) + '-wrapper"' : '') + '>' + Confluence.Templates.PageBanner.itemAnchor(soy.$$augmentMap(opt_data, {isSystemContentItem: false})) + '</li>';
};
if (goog.DEBUG) {
  Confluence.Templates.PageBanner.renderPageBannerItem.soyTemplateName = 'Confluence.Templates.PageBanner.renderPageBannerItem';
}


Confluence.Templates.PageBanner.itemAnchor = function(opt_data, opt_ignored) {
  return '<a href="' + soy.$$escapeHtml(opt_data.href) + '" title="' + soy.$$escapeHtml(opt_data.tooltip) + '" ' + ((opt_data.linkId) ? 'id="' + soy.$$escapeHtml(opt_data.linkId) + '"' : '') + ' ' + ((opt_data.styleClasses) ? 'class="' + soy.$$escapeHtml(opt_data.styleClasses) + '"' : '') + '>' + Confluence.Templates.PageBanner.itemIcon(opt_data) + ((! opt_data.isSystemContentItem) ? '<span>' + soy.$$escapeHtml(opt_data.label) + '</span>' : '') + '</a>';
};
if (goog.DEBUG) {
  Confluence.Templates.PageBanner.itemAnchor.soyTemplateName = 'Confluence.Templates.PageBanner.itemAnchor';
}


Confluence.Templates.PageBanner.itemIcon = function(opt_data, opt_ignored) {
  return '' + ((opt_data.icon) ? '<img class="page-banner-item-icon" src="' + soy.$$escapeHtml(opt_data.icon.url) + '" style="height: ' + soy.$$escapeHtml(opt_data.icon.height) + 'px; width: ' + soy.$$escapeHtml(opt_data.icon.width) + 'px;"/>' : '');
};
if (goog.DEBUG) {
  Confluence.Templates.PageBanner.itemIcon.soyTemplateName = 'Confluence.Templates.PageBanner.itemIcon';
}


Confluence.Templates.PageBanner.draftStatusDiscovery = function(opt_data, opt_ignored) {
  return '<aui-inline-dialog id="' + soy.$$escapeHtml(opt_data.elementId) + '" alignment="bottom left" open><p><strong>' + soy.$$escapeHtml('Didn\x27t publish?') + '</strong></p><p>' + soy.$$filterNoAutoescape(AJS.format('All good, we\x27\x27ve saved your changes! Find them any time under \x3ca href\x3d\x22{0}\x22\x3eRecently worked on\x3c/a\x3e. Just keep editing, and publish when you\x27\x27re ready.',opt_data.linkToMyWork)) + '</p><p><a class="aui-button">' + soy.$$escapeHtml('Okay, got it!') + '</a></p></aui-inline-dialog>';
};
if (goog.DEBUG) {
  Confluence.Templates.PageBanner.draftStatusDiscovery.soyTemplateName = 'Confluence.Templates.PageBanner.draftStatusDiscovery';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-page-banner:page-banner-common-resources', location = 'js/page-banner.js' */
define("confluence-page-banner/page-banner",["ajs","jquery","confluence/templates","confluence/legacy","skate"],function(d,c,p,q,e){function f(){var a=c("#system-content-items");0==a.children(":not(.hidden)").length?a.addClass("hidden"):a.removeClass("hidden")}function l(){g("#system-content-items a:not(.tipsy-disabled),.page-metadata-item a:not(.tipsy-disabled),.page-metadata-modification-info a.last-modified:not(tipsy-disabled),#draft-status-lozenge");c("#system-content-items a:not(.tipsy-disabled),.page-metadata-item a:not(.tipsy-disabled),.page-metadata-modification-info a.last-modified:not(tipsy-disabled),#draft-status-lozenge").click(function(a){h(c(a.target).closest("a"))});
c(window).on("click scroll resize",h)}function g(a){c(document).tooltip({live:a,gravity:"n",title:"title",delayIn:500})}function h(a){c(".tipsy").remove();a&&(a=c(a).data("tipsy"))&&(a.hoverState="out")}var k=function(a,b){a=c("#content-metadata-page-restrictions");a.removeClass("aui-iconfont-locked aui-iconfont-unlocked restricted");var e=b.hasRestrictions&&!(b.hasExplicitRestrictions||b.hasInheritedRestrictions);b.hasExplicitRestrictions||e?(a.addClass("aui-icon aui-icon-small aui-iconfont-locked restricted"),
b="Restrictions apply"):b.hasInheritedRestrictions?(a.addClass("aui-icon aui-icon-small aui-iconfont-unlocked restricted"),b="Restrictions apply"):(a.addClass("aui-icon aui-icon-small aui-iconfont-unlocked"),b=b.hasAnyExplicitRestrictions?"Restrictions apply":"Unrestricted");a.attr("title",b);g("#content-metadata-page-restrictions");f()},m=function(){f();d.bind("system-content-metadata.toggled-restrictions",k);
c("#page-metadata-banner").css("visibility","visible");l()},n=function(){d.unbind("system-content-metadata.toggled-restrictions",k)};(function(){e("system-metadata-restrictions",{type:e.types.CLASS,events:{click:function(a,b){b.preventDefault();d.trigger("system-content-metadata.open-restrictions-dialog")}},attached:m,detached:n})})()});require("confluence/module-exporter").safeRequire("confluence-page-banner/page-banner");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-page-banner:page-banner-common-resources', location = 'js/page-banner-analytics.js' */
AJS.toInit(function(c){AJS.bind("breadcrumbs.expanded",function(){AJS.trigger("analyticsEvent",{name:"breadcrumbs-expanded"})});(function(){if(AJS.Confluence.Analytics&&AJS.Confluence.Analytics.setAnalyticsSource){var a=AJS.Confluence.Analytics.setAnalyticsSource,b=c("#breadcrumbs \x3e li");a(b.filter(":not(#ellipsis)").find("a"),"breadcrumbs");a(b.filter(".hidden-crumb").find("a"),"breadcrumbs-expanded");a(b.filter(":last").find("a"),"breadcrumbs-parent");var d=c("#com-atlassian-confluence").hasClass("theme-documentation")?
"breadcrumbs-homepage":"breadcrumbs-collector";a(b.filter(".first").find("a"),d)}else AJS.log("WARNING: Could not initialise analytics for the page banner: AJS.Confluence.Analytics.setAnalyticsSource is not defined.")})()});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-feature-discovery-plugin:confluence-feature-discovery-plugin-resources', location = '/js/confluence-feature-discovery-plugin.js' */
(function(n){function k(a){void 0===d&&(d=JSON.parse(AJS.Meta.get("discovered-plugin-features")||"{}"));return a?d[a]||[]:d}function l(a,g){a=k(a);for(var f=0,e=a.length;f<e;f++)if(a[f]===g)return!0;return!1}Confluence.FeatureDiscovery={};var d,m=!1,p=WRM.data.claim("com.atlassian.confluence.plugins.confluence-feature-discovery-plugin:confluence-feature-discovery-plugin-resources.test-mode");Confluence.FeatureDiscovery.forPlugin=function(a,g){function f(b){b=parseInt(e.getItem(b),10);return isNaN(b)?
0:b}var e=Confluence.storageManager("feature-discovery."+a);g=g||3;return{addDiscoveryView:function(b){var a=f(b)+1;e.setItem(b,a)},shouldShow:function(b,c){if(c=!0===c&&p){a:{c="com.atlassian.webdriver.discovery\x3d"+a+":"+b;for(var e=document.cookie.split(";"),d=0;d<e.length;d++){for(var h=e[d];" "===h.charAt(0);)h=h.substring(1);if(-1!==h.indexOf(c)){c=!0;break a}}c=!1}c=!c}return c||m||l(a,b)?!1:f(b)>=g?(this.markDiscovered(b),!1):m=!0},markDiscovered:function(b,c){l(a,b)||AJS.safeAjax({url:AJS.contextPath()+
"/rest/feature-discovery/1.0/discovered/"+a+"/"+b,type:"POST",data:{},success:function(){k(a).unshift(b);e.removeItem(b);AJS.trigger("feature-discovered",{pluginKey:a,featureKey:b});c&&n.isFunction(c)&&c()}})},listDiscovered:function(){return k(a).slice(0)}}}})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-page-banner:status-lozenge-onboarding', location = 'js/status-lozenge-onboarding.js' */
define("confluence-page-banner/status-lozenge-onboarding",["ajs","jquery","confluence/templates","confluence/legacy"],function(d,a,e,b){d.toInit(function(){if(!b.FeatureDiscovery||!b.FeatureDiscovery.forPlugin||0===a("#draft-status-lozenge").length||0>document.referrer.indexOf("resumedraft.action")&&0>document.referrer.indexOf("editpage.action")?0:b.FeatureDiscovery.forPlugin("com.atlassian.confluence.plugins.confluence-page-banner").shouldShow("recently-work-on-contributor-lozenge")){var c=a(e.PageBanner.draftStatusDiscovery({elementId:"dope-draft-discovery",
linkToMyWork:d.contextPath()+"/#recently-worked"}));a("#draft-status-lozenge").attr("aria-controls","dope-draft-discovery");a("body").append(c);c.find(".aui-button").click(function(){c.removeAttr("open");b.FeatureDiscovery.forPlugin("com.atlassian.confluence.plugins.confluence-page-banner").markDiscovered("recently-work-on-contributor-lozenge")})}})});require("confluence/module-exporter").safeRequire("confluence-page-banner/status-lozenge-onboarding");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:view-comment', location = '/includes/js/comments.js' */
define("confluence/comments",["ajs","jquery","confluence/storage-manager","confluence/message-controller"],function(d,a,g,c){function f(b){confirm("Are you sure you want to delete the comment?")&&a.ajax({type:"DELETE",url:d.contextPath()+"/rest/api/content/"+b,contentType:"application/json",dataType:"json"}).done(function(){var a=document.location.search;-1===a.indexOf("showComments")?(a+=-1!==a.indexOf("?")?"&showComments=true":"?showComments=true",document.location.search=a):document.location.reload()}).fail(function(a){c.showError(c.parseError(a),
c.Location.FLAG)})}return{binder:{bindRemoveConfirmation:function(b){a("#comment-"+b+" .comment-action-remove a").click(function(a){a.preventDefault();f(b);return!1})}},initialiser:function(){var b=g("confluence","comments");if(a("#comments-section").length){a("#comments-section").find(".comment:odd").addClass("odd");a(".comment-action-remove a").click(function(){var b=a(this).attr("id").replace(/remove-comment-/g,"");f(b);return!1});var c=a("#addcomment.comment-text"),e=a("#comments-text-editor").find("textarea");
e.focus(function(){c.addClass("active")}).blur(function(){a.trim(e.val()).length||c.removeClass("active")}).bind("reset.default-text",function(){c.removeClass("active")});a("form[name='textcommentform']").submit(function(){var b=e.val();return!a.trim(b)?(alert("Comment text is empty. Cannot create empty comments."),!1):!0});a("#add-comment-rte").click(function(){e.hasClass("placeholded")||b.setItem("text-comment",a.trim(e.val()))});a("#addcomment #rte").length&&d.bind("init.rte",function(a,d){var c=b.getItem("text-comment");
c&&(d.editor.setContent(c),b.setItem("text-comment",""))})}}}});require("confluence/module-exporter").safeRequire("confluence/comments",function(d){require("confluence/legacy").Comments=d.binder;require("ajs").toInit(d.initialiser)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:browser', location = '/includes/js/api/browser.js' */
define("confluence/api/browser",[],function(){return function(a){function c(){return-1!==a.indexOf("Firefox/")}function b(){return-1!==a.indexOf("Edge/")}function d(){return-1!==a.indexOf("MSIE")||-1!==a.indexOf("Trident/")||b()}function e(){return-1!==a.indexOf("Chrome/")}function f(){return-1!==a.indexOf("Safari/")&&-1===a.indexOf("Chrome/")}function g(){return-1!==a.indexOf("PhantomJS")}return{isFirefox:c,notFirefox:function(){return!c()},isMSEdge:b,notMSEdge:function(){return!b()},isIE:d,notIE:function(){return!d()},
isChrome:e,notChrome:function(){return!e()},isSafari:f,notSafari:function(){return!f()},isPhantom:g,notPhantom:function(){return!g()},version:function(){if(d()){var b=a.match(/MSIE\s([\d.]+)/)||a.match(/rv:([\d.]+)/)||a.match(/Edge\/([\d.]+)/);return parseInt(b[1])}if(e())return parseInt(a.match(/Chrome\/([\d.]+)/)[1]);if(f())return parseInt(a.match(/Version\/([\d.]+)/)[1]);if(c())return parseInt(a.match(/Firefox\/([\d.]+)/)[1])},friendlyName:function(){if(b())return"MSEdge";if(d())return"IE";if(e())return"Chrome";
if(f())return"Safari";if(c())return"Firefox"}}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.editor:page-editor-message', location = 'editor/page-editor-message.js' */
define("confluence-editor/editor/page-editor-message",["jquery","ajs","aui/flag","document","underscore"],function(f,i,g,h,d){var b=Object.create(null),e=Object.create(null);h.addEventListener("aui-flag-close",function(a){a&&a.target&&(b=d.filter(Object.keys(b),function(c){return!f(a.target).find("span").hasClass(c)}))});return{handleMessage:function(a,c,d){b[a]||(e[a]?delete e[a]:c&&(b[a]=g({title:c.title?c.title:"",type:c.type,close:c.close?c.close:"manual",persistent:!1,body:"<span class='"+a+
"'>"+c.message+"</span>"}),d&&d()))},closeMessages:function(a){d.each(a,function(a){b[a]&&(b[a].close(),delete b[a])})},isDisplayed:function(a){return a in b},displayedErrors:function(){return Object.keys(b)},suppressMessage:function(a){e[a]={}}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:page-message', location = '/includes/js/page-message.js' */
define("confluence/page-message","ajs jquery confluence/meta window confluence/api/browser confluence-editor/editor/page-editor-message".split(" "),function(a,h,d,f,i,j){var e={},g=new i(f.navigator.userAgent);e._getQueryString=function(){return f.location.search};e.displayPageMessage=function(){var c=e._getQueryString(),b=d.get("editing-user");if(c.indexOf("editingLocked")!==-1&&b)a.MessageHandler.flag({type:"info",title:"Unable to edit",body:a.format("Collaborative editing is offline right now, and {0} is editing this page. Try again in a few minutes.",
a.escapeHtml(b)),close:"manual"});else if(c.indexOf("editingFailed")!==-1)a.MessageHandler.flag({type:"info",title:"Unable to edit",body:"Editing is unavailable right now. Try again in a few minutes.",close:"manual"});else if(c.indexOf("userLimitReached")!==-1){j.handleMessage("collab.edit.user.limit.reached",{type:"warning",title:"You can\u0027t edit this page right now",message:"\u003cp\u003eThis page is so popular, you\u0027ve reached the maximum number of simultaneous editors.\u003c/p\u003e\u003cp\u003eTry again when it\u0027s not so busy.\u003c/p\u003e",close:"manual"});a.Confluence.Analytics.publish("collab.edit.user.limit.reached",
{browserName:g.friendlyName(),browserVersion:g.version(),pageId:d.get("page-id"),editMode:"slow",numEditors:d.get("max-number-editors")})}else if(c.indexOf("spaceEditingRestriction")!==-1){b=d.get("space-key");c=d.get("content-type")==="blogpost"?"You can\u0027t edit blog posts in this space":"You can\u0027t edit pages in this space";b=d.get("context-path")+"/spaces/viewspacesummary.action?key="+encodeURIComponent(b);b=d.get("content-type")==="blogpost"?a.format("Ask a {0}space admin{1}, or the person who invited you, to give you \u0027\u0027add blog\u0027\u0027 permission in this space.",
'<a href="'+b+'">',"</a>"):a.format("Ask a {0}space admin{1}, or the person who invited you, to give you \u0027\u0027add page\u0027\u0027 permission in this space.",'<a href="'+b+'">',"</a>");c=a.MessageHandler.flag({type:"info",title:c,body:b,close:"manual"});h(c).addClass("spaceEditingRestriction")}};return e});require("confluence/module-exporter").safeRequire("confluence/page-message",function(a){require("ajs").toInit(a.displayPageMessage)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:legacy-scroll-to-hash-anchors', location = '/includes/js/legacy-scroll-to-hash-anchors.js' */
define("confluence/legacy-scroll-to-hash-anchors",["window","confluence/api/logger"],function(b,f){function g(){e=!0}function h(){var a,c;if(!(e||0<b.pageYOffset))for(c=b.location.hash;c;){var d=c;a=d.indexOf("-");d=d.indexOf("#");a=0>a?d:0>d?a:Math.min(d,a);if(0>a)break;c=c.substring(0,a).concat(c.substring(a+1));if(a=b.document.getElementById(c)){f.log("Legacy anchor found. Scrolling the page to ",a);b.location.hash="#"+c;break}}}var e=!1;return{bindToWindowEvents:function(){b.addEventListener("load",
h,{passive:!0});b.addEventListener("scroll",g,{passive:!0})}}});require("confluence/module-exporter").safeRequire("confluence/legacy-scroll-to-hash-anchors",function(b){b.bindToWindowEvents()});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-comment-page', location = 'soy/comments.soy' */
// This file was automatically generated from comments.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Comments.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Comments == 'undefined') { Confluence.Templates.Comments = {}; }


Confluence.Templates.Comments.displayReplyEditorLoadingContainer = function(opt_data, opt_ignored) {
  return '<ol class="comment-threads"><li class="comment-thread">' + Confluence.Templates.Comments.displayCommentEditorCommon({comment: {ownerId: opt_data.contentId, parentId: opt_data.parentCommentId}, commenter: opt_data.commenter, state: 'loading', mode: 'reply'}) + '</li></ol>';
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.displayReplyEditorLoadingContainer.soyTemplateName = 'Confluence.Templates.Comments.displayReplyEditorLoadingContainer';
}


Confluence.Templates.Comments.displayTopLevelCommentEditorPlaceholder = function(opt_data, opt_ignored) {
  return '' + Confluence.Templates.Comments.displayCommentEditorCommon({comment: {ownerId: opt_data.contentId}, commenter: opt_data.commenter, state: 'placeholder', mode: 'add'});
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.displayTopLevelCommentEditorPlaceholder.soyTemplateName = 'Confluence.Templates.Comments.displayTopLevelCommentEditorPlaceholder';
}


Confluence.Templates.Comments.displayEditEditorContainer = function(opt_data, opt_ignored) {
  return '' + Confluence.Templates.Comments.displayCommentEditorCommon({comment: {ownerId: opt_data.contentId, id: opt_data.commentId}, commenter: opt_data.commenter, state: 'placeholder', mode: 'edit'});
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.displayEditEditorContainer.soyTemplateName = 'Confluence.Templates.Comments.displayEditEditorContainer';
}


Confluence.Templates.Comments.editorLoadErrorMessage = function(opt_data, opt_ignored) {
  return '<p>' + soy.$$escapeHtml(opt_data.message) + '</p><p><a href="' + soy.$$escapeHtml(opt_data.fallbackUrl) + '">' + soy.$$escapeHtml('Try again') + '</a></p>';
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.editorLoadErrorMessage.soyTemplateName = 'Confluence.Templates.Comments.editorLoadErrorMessage';
}


Confluence.Templates.Comments.displayCommentEditorCommon = function(opt_data, opt_ignored) {
  var output = '<div class="quick-comment-container comment ' + soy.$$escapeHtml(opt_data.mode) + '">' + Confluence.Templates.Comments.userLogo({userInfo: opt_data.commenter}) + '<div class="quick-comment-body"><div class="quick-comment-loading-container" style="display:' + ((opt_data.state == 'loading') ? 'block' : 'none') + ';"></div><div id="editor-messages"></div><div id="any-messages"></div><form style="display:' + ((opt_data.state == 'loading') ? 'none' : 'block') + ';" class="quick-comment-form aui" method="post" ';
  switch (opt_data.mode) {
    case 'add':
      output += 'name="inlinecommentform" action="' + soy.$$escapeHtml("/wiki") + '/pages/doaddcomment.action?pageId=' + soy.$$escapeHtml(opt_data.comment.ownerId) + '"';
      break;
    case 'edit':
      output += 'name="editcommentform" action="' + soy.$$escapeHtml("/wiki") + '/pages/doeditcomment.action?pageId=' + soy.$$escapeHtml(opt_data.comment.ownerId) + '&commentId=' + soy.$$escapeHtml(opt_data.comment.id) + '"';
      break;
    case 'reply':
      output += 'name="threadedcommentform"  action="' + soy.$$escapeHtml("/wiki") + '/pages/doaddcomment.action?pageId=' + soy.$$escapeHtml(opt_data.comment.ownerId) + '&parentId=' + soy.$$escapeHtml(opt_data.comment.parentId) + '"';
      break;
  }
  output += ' >' + ((opt_data.mode == 'add') ? '<div title="' + soy.$$escapeHtml('Add a Comment') + '" role="button" tabindex="0" aria-label="' + soy.$$escapeHtml('Add a Comment') + '" class="quick-comment-prompt"><span>' + soy.$$escapeHtml('Write a comment...') + '</span></div>' : '') + '</form></div></div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.displayCommentEditorCommon.soyTemplateName = 'Confluence.Templates.Comments.displayCommentEditorCommon';
}


Confluence.Templates.Comments.userLogo = function(opt_data, opt_ignored) {
  return '<p class="comment-user-logo">' + ((opt_data.userInfo.userName == null) ? '<img class="userLogo logo anonymous" src="' + soy.$$escapeHtml("/wiki/s/nb8bdl/8703/51k4y0/_") + '/images/icons/profilepics/anonymous.svg" alt="' + soy.$$escapeHtml('User icon') + ': ' + soy.$$escapeHtml('Anonymous') + '" title="' + soy.$$escapeHtml('Anonymous') + '">' : (opt_data.userInfo.profilePicture.isDefault) ? '<a class="userLogoLink" data-username="' + soy.$$escapeHtml(opt_data.userInfo.userName) + '" href="' + soy.$$escapeHtml("/wiki") + '/users/profile/editmyprofilepicture.action" title="' + soy.$$escapeHtml('Add a picture of yourself') + '"><img class="userLogo logo defaultLogo" src="' + soy.$$escapeHtml("/wiki/s/nb8bdl/8703/51k4y0/_") + '/images/icons/profilepics/add_profile_pic.svg" alt="' + soy.$$escapeHtml('User icon') + ': ' + soy.$$escapeHtml('Add a picture of yourself') + '"></a>' : '<a class="userLogoLink" data-username="' + soy.$$escapeHtml(opt_data.userInfo.userName) + '" href="' + Confluence.Templates.User.userLinkUrl({username: opt_data.userInfo.userName}) + '"><img class="userLogo logo" src="' + soy.$$escapeHtml(opt_data.userInfo.profilePicture.path) + '" alt="' + soy.$$escapeHtml('User icon') + ': ' + soy.$$escapeHtml(opt_data.userInfo.userName) + '" title="' + soy.$$escapeHtml(opt_data.userInfo.userName) + '"></a>') + '</p>';
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.userLogo.soyTemplateName = 'Confluence.Templates.Comments.userLogo';
}


Confluence.Templates.Comments.displayComment = function(opt_data, opt_ignored) {
  return '' + ((opt_data.comment.parentId == 0 && opt_data.firstReply) ? '<div id="comments-section" class="pageSection group"><div class="section-header"><h2 id="comments-section-title" class="section-title">' + soy.$$escapeHtml('1 Comment') + '</h2>' + Confluence.Templates.Comments.commentThread({comment: opt_data.comment, commenter: opt_data.commenter, highlight: opt_data.highlight, topLevel: true}) + '</div></div>' : (opt_data.firstReply) ? Confluence.Templates.Comments.commentThread({comment: opt_data.comment, commenter: opt_data.commenter, highlight: opt_data.highlight, topLevel: false}) : Confluence.Templates.Comments.commentThreadItem({comment: opt_data.comment, commenter: opt_data.commenter, highlight: true}));
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.displayComment.soyTemplateName = 'Confluence.Templates.Comments.displayComment';
}


Confluence.Templates.Comments.commentThread = function(opt_data, opt_ignored) {
  return '<ol class="comment-threads' + ((opt_data.topLevel) ? ' top-level" id="page-comments' : '') + '">' + Confluence.Templates.Comments.commentThreadItem(opt_data) + '</ol>';
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.commentThread.soyTemplateName = 'Confluence.Templates.Comments.commentThread';
}


Confluence.Templates.Comments.commentThreadItem = function(opt_data, opt_ignored) {
  return '<li id="comment-thread-' + soy.$$escapeHtml(opt_data.comment.id) + '" class="comment-thread">' + Confluence.Templates.Comments.commentView(opt_data) + '</li>';
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.commentThreadItem.soyTemplateName = 'Confluence.Templates.Comments.commentThreadItem';
}


Confluence.Templates.Comments.commentView = function(opt_data, opt_ignored) {
  return '<div class="comment' + ((opt_data.highlight == true) ? ' focused' : '') + '" id="comment-' + soy.$$escapeHtml(opt_data.comment.id) + '">' + Confluence.Templates.Comments.userLogo({userInfo: opt_data.commenter}) + '<div class="comment-header"><h4 class="author">' + ((opt_data.commenter.userName == null) ? soy.$$escapeHtml('Anonymous') : '<a href="' + soy.$$escapeHtml("/wiki") + '/display/~' + soy.$$escapeUri(opt_data.commenter.userName) + '" class="url fn confluence-userlink" data-username="' + soy.$$escapeHtml(opt_data.commenter.userName) + '">' + soy.$$escapeHtml(opt_data.commenter.displayName) + '</a>') + '</h4></div><div class="comment-body"><div class="comment-content wiki-content">' + soy.$$filterNoAutoescape(opt_data.comment.html) + '</div><div class="comment-actions">' + Confluence.Templates.Comments.displayCommentActions({section: 'secondary', actions: opt_data.comment.secondaryActions, commentId: opt_data.comment.id}) + Confluence.Templates.Comments.displayCommentActions({section: 'primary', actions: opt_data.comment.primaryActions, commentId: opt_data.comment.id}) + '</div></div></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.commentView.soyTemplateName = 'Confluence.Templates.Comments.commentView';
}


Confluence.Templates.Comments.displayCommentActions = function(opt_data, opt_ignored) {
  var output = '<ul class="comment-actions-' + soy.$$escapeHtml(opt_data.section) + '">';
  if (opt_data.actions != null) {
    var itemList213 = opt_data.actions;
    var itemListLen213 = itemList213.length;
    for (var itemIndex213 = 0; itemIndex213 < itemListLen213; itemIndex213++) {
      var itemData213 = itemList213[itemIndex213];
      output += '<li ' + ((itemData213.style != null) ? ' class="' + soy.$$escapeHtml(itemData213.style) + '"' : '') + '><a ' + ((itemData213.tooltip != null) ? ' title="' + soy.$$escapeHtml(itemData213.tooltip) + '"' : '') + ' href="' + soy.$$escapeHtml(itemData213.url) + '" ' + ((itemData213.id) ? ' id="' + soy.$$escapeHtml(itemData213.id) + '-' + soy.$$escapeHtml(opt_data.commentId) + '"' : '') + '><span>' + soy.$$escapeHtml(itemData213.label) + '</span></a></li>\n';
    }
  }
  output += '</ul>';
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Comments.displayCommentActions.soyTemplateName = 'Confluence.Templates.Comments.displayCommentActions';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-comment-page', location = 'jscripts/comment-display-manager.js' */
define("confluence-quick-edit/comment-display-manager",["ajs","confluence/legacy","confluence/templates","jquery"],function(g,j,h,c){var i={_updateCommentSectionTitle:function(){var a=c("#comments-section-title");if(0!==a.length){var b=this.commentCount();1===b?a.text("1 Comment"):a.text(g.format("{0} Comments",b))}},addComment:function(a,b,f,d){a={comment:b,commenter:a,highlight:f,context:{contextPath:g.Meta.get("context-path"),staticResourceUrlPrefix:g.Meta.get("static-resource-url-prefix")}};
if(this.hasComments()){if(0===b.parentId)a.firstReply=!1,f=c("#comments-section").find(".comment-threads.top-level");else{var f=c("#comment-thread-"+b.parentId),e=f.children(".commentThreads");e.length?(a.firstReply=!1,f=e):a.firstReply=!0}d||this.clearFocus();d=c(h.Comments.displayComment(a));d.addClass("fadeInComment");f.append(d);this._updateCommentSectionTitle()}else a.firstReply=!0,d=c(h.Comments.displayComment(a)),d.addClass("fadeInComment"),c("#comments-section").prepend(d);j.Comments.bindRemoveConfirmation(b.id);
b=this.getCommentNode(b.id);b.scrollToElement();return b},addOrUpdateComment:function(a,b,c,d){var e=this.getCommentNode(b.id);return e?(e.find(".comment-content").html(b.html),d||this.clearFocus(),c&&e.addClass("focused"),e.addClass("fadeInComment"),e.scrollToElement(),e):this.addComment.apply(this,arguments)},isVisible:function(){return!c("#page-comments").hasClass("hidden")},hasComments:function(){return 0<this.commentCount()},commentCount:function(){return c("#comments-section .comment").not(".quick-comment-container").length},
clearFocus:function(){c(".comment").removeClass("focused")},getCommentNode:function(a){a=c("#comment-"+a);return!a.length?null:a},getParentId:function(a){a=i.getCommentNode(a);return null!=a&&(a=a.closest("div.comment"),a.length)?(a=a.attr("id"),a=/\d+/.exec(a),parseInt(a)):0}};return i});require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/comment-display-manager","Confluence.CommentDisplayManager");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-comment-page', location = 'jscripts/scroll-util.js' */
define("confluence-quick-edit/scroll-util",["window","document","jquery"],function(a,c,e){return{scrollToElement:function(){this.scrollWindowToElement()||this.scrollOverflowContainerToElement();return this},scrollWindowToElement:function(){function b(){return"pageYOffset"in a?a.pageYOffset:c.documentElement.scrollTop}var f=b(),d;if("number"===typeof a.innerWidth)d=a.innerHeight;else if(c.documentElement&&c.documentElement.clientWidth)d=c.documentElement.clientHeight;else return this[0].scrollIntoView(!1),
!0;var g=this.offset().top,h=this.height();return g+h+40>f+d?(this[0].scrollIntoView(!1),a.scrollBy(0,40),f!=b()):!0},scrollOverflowContainerToElement:function(){var b=null;this.parents().each(function(){var a=e(this).css("overflow");if("auto"==a||"scroll"==a)return b=e(this),!1});if(!b)return!1;var a=b.height(),d=this.offset().top,c=this.height(),a=a-(d+c+40);0>a&&(b[0].scrollTop-=a);return!0}}});require("confluence/module-exporter").safeRequire("confluence-quick-edit/scroll-util",function(a){require("jquery").fn.extend(a)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-core', location = 'utils/quick-reload-timer.js' */
define("confluence-quick-reload/utils/quick-reload-timer",["jquery","underscore"],function(c,d){function a(b,a){this.options=c.extend({min:3E4,max:36E5,inactivity:12E4},a);d.bindAll(this,"start","stop","_now","_timeSinceLastSeen","_setActive","_setActivityTrigger","_clamp","_intervalMultiplier");this.callback=b;this._timer=null;this._setActivityTrigger(this._setActive)}a.prototype.start=function(b){this.lastSeenTimestamp=this._now();this.stop();b&&this.callback();var a=function(){this.callback();
c.call(this)},c=function(){this.stop();this._timer=setTimeout(d.bind(a,this),this.options.min*this._intervalMultiplier())};c.call(this)};a.prototype.stop=function(){null!==this._timer&&(clearTimeout(this._timer),this._timer=null)};a.prototype._setActive=function(){if(null!==this._timer){var b=this._timeSinceLastSeen()>this.options.inactivity;this.lastSeenTimestamp=this._now();b&&this.start(!0)}};a.prototype._setActivityTrigger=function(b){document.addEventListener("visibilitychange",function(){document.hidden||
b()},!1);c(window).focus(b);c("body").click(b);c(window).scroll(b)};a.prototype._now=function(){return(new Date).getTime()};a.prototype._timeSinceLastSeen=function(){return this._now()+1-this.lastSeenTimestamp};a.prototype._clamp=function(b,a,c){a=Math.max(Math.min(a,c),b);return isNaN(a)?b:a};a.prototype._intervalMultiplier=function(){var a=(document.hidden?3:1)*Math.ceil(this._timeSinceLastSeen()/this.options.inactivity);return this._clamp(1,this.options.max/this.options.min,a)};return a});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-core', location = 'main/quick-reload-manager.js' */
define("confluence-quick-reload/main/quick-reload-manager","underscore jquery ajs confluence/api/constants confluence-quick-reload/utils/quick-reload-timer confluence/meta confluence/api/logger".split(" "),function(f,e,m,h,k,g,l){function a(){this.enabled=!1;this.handlers=[];this.lastFetchTime=e('meta[name\x3d"confluence-request-time"]').attr("content")||(new Date).getTime();this.timer=null;f.bindAll(this,"addHandler","removeHandler","update","enable","disable","onUpdateSuccess","onUpdateError")}
a.prototype.addHandler=function(b){var a=!1,d;for(d=0;d<this.handlers.length;d++)this.handlers[d]===b&&(a=!0);!0!==a&&this.handlers.push(b)};a.prototype.removeHandler=function(b){var a;if(b)for(a=0;a<this.handlers.length;a++)if(this.handlers[a]===b){this.handlers.splice(a,1);break}};a.prototype.update=function(){var a=void 0!==g.get("page-id")?g.get("page-id"):"0";e("body").hasClass("contenteditor")?this.disable():e.ajax({type:"GET",url:h.CONTEXT_PATH+"/rest/quickreload/latest/"+a+"?since\x3d"+encodeURIComponent(this.lastFetchTime),
dataType:"json"}).done(this.onUpdateSuccess).fail(this.onUpdateError)};a.prototype.enable=function(){this.enabled||(null===this.timer&&(this.timer=new k(this.update)),this.timer.start(),this.enabled=!0)};a.prototype.disable=function(){null!==this.timer&&this.timer.stop();this.enabled=!1};a.prototype.isEnabled=function(){return this.enabled};a.prototype.onUpdateSuccess=function(a,e,d){204!==d.status&&(this.lastFetchTime=a.time,f.map(this.handlers,function(b){var c=a[b.property];Array.isArray(c)||(c=
[c]);c=b.filterNewResults(b.results,c);0<c.length&&(b.results=b.results.concat(c),b.render(c))},this))};a.prototype.onUpdateError=function(a){if(a={404:"not found - the plugin has been probably been removed or disabled from Confluence",500:"generic server error",503:"service unavailable",504:"gateway timeout"}[(a||{}).status])this.disable(),l.error('Quick comment reload plugin has been disabled in this page due to a server error: "'+a+'". Please refresh the page to get it back.')};return new a});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-read-only-mode-handler', location = 'handlers/quick-reload-read-only-mode.js' */
define("confluence-quick-reload/handlers/quick-reload-read-only-mode",["ajs","jquery","confluence/message-controller","confluence/meta","confluence/api/event"],function(f,k,a,e,g){return{results:[],property:"readOnlyMode",filterNewResults:function(a,c){return 0<c.length&&"string"!==typeof c[0]?c:[]},render:function(b){b=b[0];var c=e.get("access-mode"),d=b.isEnabled?"READ_ONLY":"READ_WRITE",h="READ_ONLY"===e.get("render-mode")?"We\u0027re back! \u003ca onclick=\u0022location.reload()\u0022\u003eRefresh\u003c/a\u003e the page to edit and make changes.":"We\u0027re back! You can now edit and make changes.";
if(!a.hasSuccess(a.Location.BANNER)||b.isEnabled)e.set("access-mode",d),d!==c&&g.trigger("qr:access-mode",{accessMode:d}),this.shouldDisplayError(b)?a.hasErrors(a.Location.BANNER)||this.showError(b.bannerMessage):"READ_ONLY"===c&&"READ_WRITE"===d?this.showSuccess(h):this.clearAll()},clearAll:function(){a.clearErrors(a.Location.ALL);a.clearSuccess(a.Location.ALL)},showError:function(b){a.clearSuccess(a.Location.ALL);a.showError(b,a.Location.BANNER)},showSuccess:function(b){a.clearErrors(a.Location.ALL);
a.showSuccess(b,a.Location.BANNER,{closeable:!0})},shouldDisplayError:function(a){return a.isBannerMessageOn||a.isEnabled}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-resources', location = 'main/quick-reload.soy' */
// This file was automatically generated from quick-reload.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace QuickReload.Templates.
 */

if (typeof QuickReload == 'undefined') { var QuickReload = {}; }
if (typeof QuickReload.Templates == 'undefined') { QuickReload.Templates = {}; }


QuickReload.Templates.pageAttachments = function(opt_data, opt_ignored) {
  return '' + QuickReload.Templates.container({content: '<div class="qr-notice-authors"></div><div class="qr-notice-summary qr-notice-summary-attachments">' + aui.buttons.button({text: 'Reload page', type: 'text', extraClasses: 'qr-notice-show aui-button-link'}) + '</div>'});
};
if (goog.DEBUG) {
  QuickReload.Templates.pageAttachments.soyTemplateName = 'QuickReload.Templates.pageAttachments';
}


QuickReload.Templates.pageEdit = function(opt_data, opt_ignored) {
  return '' + QuickReload.Templates.container({content: '<div class="qr-notice-authors">' + QuickReload.Templates.noticeHeader({users: opt_data.pageEditors}) + '</div><div class="qr-notice-summary">' + aui.buttons.button({text: 'Reload page', type: 'text', extraClasses: 'qr-notice-show aui-button-link'}) + QuickReload.Templates.summaryText({users: opt_data.pageEditors}) + '</div>'});
};
if (goog.DEBUG) {
  QuickReload.Templates.pageEdit.soyTemplateName = 'QuickReload.Templates.pageEdit';
}


QuickReload.Templates.pageComment = function(opt_data, opt_ignored) {
  return '' + QuickReload.Templates.container({content: '<div class="qr-notice-authors">' + QuickReload.Templates.noticeHeader({users: opt_data.commentUsers}) + ((opt_data.commentUsers.length == 1) ? '<span class="qr-notice-text">"' + soy.$$escapeHtml(opt_data.noticeText) + '"</span>' : '') + '</div><div class="qr-notice-summary">' + aui.buttons.button({text: '' + ((opt_data.commentUsers.length > 1) ? soy.$$escapeHtml('Show all') : soy.$$escapeHtml('Show')), type: 'text', extraClasses: 'qr-notice-show aui-button-link'}) + QuickReload.Templates.summaryText({users: opt_data.commentUsers}) + '</div>'});
};
if (goog.DEBUG) {
  QuickReload.Templates.pageComment.soyTemplateName = 'QuickReload.Templates.pageComment';
}


QuickReload.Templates.inlineComment = function(opt_data, opt_ignored) {
  return '' + QuickReload.Templates.container({content: '<div class="qr-notice-authors">' + QuickReload.Templates.noticeHeader({users: opt_data.user}) + '<span class="qr-notice-text">"' + soy.$$escapeHtml(opt_data.noticeText) + '"</span></div><div class="qr-notice-summary">' + aui.buttons.button({text: '' + ((opt_data.reloadRequired) ? soy.$$escapeHtml('Reload page') : soy.$$escapeHtml('Show')), type: 'text', extraClasses: 'qr-notice-show aui-button-link', extraAttributes: ['data-reload-required', opt_data.reloadRequired]}) + QuickReload.Templates.summaryText({users: opt_data.user}) + '</div>'});
};
if (goog.DEBUG) {
  QuickReload.Templates.inlineComment.soyTemplateName = 'QuickReload.Templates.inlineComment';
}


QuickReload.Templates.container = function(opt_data, opt_ignored) {
  return '<div class="qr-container">' + soy.$$filterNoAutoescape(opt_data.content) + '</div>';
};
if (goog.DEBUG) {
  QuickReload.Templates.container.soyTemplateName = 'QuickReload.Templates.container';
}


QuickReload.Templates.noticeHeader = function(opt_data, opt_ignored) {
  var output = '';
  var userList77 = opt_data.users;
  var userListLen77 = userList77.length;
  for (var userIndex77 = 0; userIndex77 < userListLen77; userIndex77++) {
    var userData77 = userList77[userIndex77];
    output += (userIndex77 < 8) ? aui.avatar.avatar({size: 'small', avatarImageUrl: userData77.profilePicture.path, title: userData77.displayName != '' ? userData77.displayName : 'Anonymous', extraClasses: 'qr-author-avatar'}) : '';
  }
  return output;
};
if (goog.DEBUG) {
  QuickReload.Templates.noticeHeader.soyTemplateName = 'QuickReload.Templates.noticeHeader';
}


QuickReload.Templates.summaryText = function(opt_data, opt_ignored) {
  var output = '';
  var lastModifier__soy87 = opt_data.users[0];
  var others__soy88 = opt_data.users.length - 1;
  output += '<span class="qr-notice-summary-text">' + soy.$$escapeHtml('by') + ' ' + Confluence.Templates.User.usernameLink({canView: false, username: lastModifier__soy87.userName, fullName: lastModifier__soy87.displayName != '' ? lastModifier__soy87.displayName : 'Anonymous'}) + ((others__soy88 == 1) ? ' ' + soy.$$escapeHtml('and 1 other') : '') + ((others__soy88 > 1) ? ' ' + soy.$$escapeHtml(AJS.format('and {0} others',others__soy88)) : '') + '</span>';
  return output;
};
if (goog.DEBUG) {
  QuickReload.Templates.summaryText.soyTemplateName = 'QuickReload.Templates.summaryText';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-resources', location = 'utils/quick-reload-count.js' */
define("confluence-quick-reload/utils/quick-reload-count",[],function(){function a(){this.count=0}var b=document.title;a.prototype.getCount=function(){return this.count};a.prototype.setCount=function(a){this.count=a;this._displayCount()};a.prototype._displayCount=function(){document.title=(0<this.count?"("+this.count+") ":"")+b};return new a});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-resources', location = 'handlers/quick-reload-comments.js' */
define("confluence-quick-reload/handlers/quick-reload-comments","ajs underscore jquery confluence/flag confluence/api/event confluence-quick-reload/utils/quick-reload-count confluence/legacy".split(" "),function(l,c,e,m,n,h,b){function p(a){a=c.clone(a);a.reverse();a=c.uniq(a,function(a){return a.commenter.userName});return c.map(a,function(a){return a.commenter})}function q(a){b&&b.CommentDisplayManager&&(b.CommentDisplayManager.clearFocus(),c.map(a,function(a){var c=null!==b.CommentDisplayManager.getCommentNode(g(a));
a=b.CommentDisplayManager.addOrUpdateComment(a.commenter,a.comment,!0,!0);b.Likes&&!c&&(b.Likes.appendAction(e(a)),b.Likes.updateComment(e(a),{}))}),b.CommentDisplayManager._updateCommentSectionTitle(),n.trigger("ic-jim-async-supported"))}function g(a){return a.comment.id}var d,f=[];return{results:[],property:"comments",ignoreOnce:function(a){f.push(a)},filterNewResults:function(a,b){if(0===b.length)return b;var d=c.map(a,g);return c.filter(b,function(a){if(a.comment.isInlineComment)return!1;a=g(a);
if(c.contains(d,a))return!1;a=e.inArray(a,f);return 0<=a?(f.splice(a,1),!1):!0})},render:function(a){h.setCount(h.getCount()+a.length);a={close:"manual",type:"info",extraClasses:"qr-flag",fifo:!0,stack:"quickreload"};var b=1<this.results.length?l.format("{0} New comments",this.results.length):"New comment",k=p(this.results),g=e("\x3cdiv\x3e\x3c/div\x3e").html(this.results[this.results.length-1].comment.html).text();k=QuickReload.Templates.pageComment({commentUsers:k,
noticeText:g});var f=this;void 0===d||"true"===d.getAttribute("aria-hidden")?(d=new m(e.extend({},{body:k,title:b},a)),e(d).on("click",".qr-notice-show",c.bind(function(){q(f.results);d.close()},this)),e(d).on("aui-flag-close",c.bind(function(){h.setCount(h.getCount()-f.results.length);f.results=[]},this))):(e(d).find(".qr-container").replaceWith(k),e(d).find(".title").text(b))}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-resources', location = 'handlers/quick-reload-inline-comments.js' */
define("confluence-quick-reload/handlers/quick-reload-inline-comments","underscore jquery ajs confluence/flag confluence/meta confluence/api/event confluence-quick-reload/utils/quick-reload-count".split(" "),function(c,e,l,u,r,t,h){function n(a,b,c,f,d){a=QuickReload.Templates.inlineComment({user:[a.commenter],noticeText:v(a),reloadRequired:b});d=1<d?"New inline comments":"New inline comment";d=new u(e.extend({},{title:d,body:a},{close:"manual",
type:"info",extraClasses:"qr-flag",fifo:!0,stack:"quickreload"}));e(d).find(".qr-notice-show").click(c);b||e(d).find(".qr-notice-show").click(d.close);e(d).on("aui-flag-close",f)}function v(a){return a&&a.comment&&e("\x3cdiv\x3e\x3c/div\x3e").text(a.comment.html).text()}function m(a){return 0===a.comment.parentId}function k(a){return m(a)?a.comment.id:a.comment.parentId}return{results:[],property:"comments",filterNewResults:function(a,b){b=c.clone(b);b=c.filter(b,function(a){return a.comment.isInlineComment&&
a.commenter.userName!==r.get("remote-user")});if(0<b.length){var e=c.map(a,k);a=c.filter(b,m);b=c.filter(b,function(a){return!m(a)});var f=c.map(a,k),d=c.map(b,k),h=c.difference(d,f),g=[];b=c.filter(b,function(a){return-1!==h.indexOf(a.comment.parentId)&&-1===g.indexOf(a.comment.parentId)?(g.push(a.comment.parentId),!0):!1});a=a.concat(b);b=c.filter(a,function(a){return-1===e.indexOf(k(a))})}return b},render:function(a){h.setCount(h.getCount()+a.length);var b=this;c.each(a,function(a){var f=k(a),
d=function(){t.trigger("qr:show-new-thread",f)},l=function(){var a=r.Links.canonical(),b=-1===a.indexOf("?")?"?":"\x26";window.location=a+b+"focusedCommentId\x3d"+f+"#comment-"+f},g=function(){b.results=c.filter(b.results,function(a){return k(a)!==f});h.setCount(h.getCount()-1)};g=c.bind(g,this);var p=this.results.length;if(m(a)){var q=new e.Deferred;q.fail(function(a){n(a,!0,l,g,p)});q.done(function(a){n(a,!1,d,g,p)});t.trigger("qr:add-new-highlight",[a,q])}else n(a,!1,d,g,p)},this)}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-resources', location = 'handlers/quick-reload-page.js' */
define("confluence-quick-reload/handlers/quick-reload-page",["underscore","jquery","ajs","confluence/flag","confluence-quick-reload/utils/quick-reload-count"],function(d,c,g,l,e){function m(a){a=d.clone(a);a.reverse();a=d.uniq(a,function(a){return a.editor.userName});return d.map(a,function(a){return a.editor})}var b;return{results:[],property:"page",filterNewResults:function(a,b){return 0<b.length&&"string"!==typeof b[0]?b:[]},render:function(a){e.setCount(e.getCount()+a.length);a={close:"manual",
type:"info",extraClasses:"qr-flag",fifo:!0,stack:"quickreload"};var h=1<this.results.length?g.format("{0} new edits",this.results.length):"New page edits",f=m(this.results);f=QuickReload.Templates.pageEdit({pageEditors:f});var k=this;void 0===b||"true"===b.getAttribute("aria-hidden")?(b=new l(c.extend({},{body:f,title:h},a)),c(b).on("click",".qr-notice-show",function(){c(this).prop("disabled",!0).prepend('\x3cspan class\x3d"aui-icon aui-icon-wait"\x3e\x3c/span\x3e\x26nbsp;');
window.location.reload()}),c(b).on("aui-flag-close",d.bind(function(){e.setCount(e.getCount()-k.results.length);k.results=[]},this))):(c(b).find(".qr-container").replaceWith(f),c(b).find(".title").text(h))}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-resources', location = 'handlers/quick-reload-attachments.js' */
define("confluence-quick-reload/handlers/quick-reload-attachments",["underscore","jquery","ajs","confluence/flag","confluence-quick-reload/utils/quick-reload-count"],function(h,a,k,l,d){var b;return{results:[],property:"attachmentsCount",filterNewResults:function(b,a){const c=a[0];return"number"===typeof c&&0<c?(b.length=0,a):[]},render:function(c){c=c[0]||0;d.setCount(d.getCount()+c);c={close:"manual",type:"info",extraClasses:"qr-flag",fifo:!0,stack:"quickreload"};const e="Files on this page have changed",
f=QuickReload.Templates.pageAttachments({});if(void 0===b||"true"===b.getAttribute("aria-hidden")){b=new l(a.extend({},{body:f,title:e},c));a(b).on("click",".qr-notice-show",function(){a(this).prop("disabled",!0).prepend('\x3cspan class\x3d"aui-icon aui-icon-wait"\x3e\x3c/span\x3e\x26nbsp;');window.location.reload()});const g=this;a(b).on("aui-flag-close",h.bind(function(){d.setCount(d.getCount()-(g.results[0]||0));g.results=[]},this))}else a(b).find(".qr-container").replaceWith(f),a(b).find(".title").text(e)}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-bootstrap', location = 'main/quick-reload-main.js' */
require("ajs confluence/dark-features confluence-quick-reload/main/quick-reload-manager confluence-quick-reload/handlers/quick-reload-comments confluence-quick-reload/handlers/quick-reload-inline-comments confluence-quick-reload/handlers/quick-reload-page confluence-quick-reload/handlers/quick-reload-read-only-mode confluence-quick-reload/handlers/quick-reload-attachments confluence/legacy".split(" "),function(b,d,a,c,e,f,g,h,k){b.toInit(function(){if(d.isEnabled("quickreload.disabled")||!k.CommentDisplayManager?
0:void 0!==b.Meta.get("page-id"))a.addHandler(c),a.addHandler(e),a.addHandler(f),a.addHandler(h),a.addHandler(g),a.enable(),b.bind("page.commentAddedOrUpdated",function(a,b){c.ignoreOnce(b.commentId)})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-sortable-tables:sortable-table-hooks', location = 'js/hooks.js' */
define("confluence-sortable-tables/hooks",[],function(){var a=[];return{onBeforeInit:function(b){a.push(b)},beforeInitHooks:function(){return a}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-content-report-plugin:resources', location = 'com/atlassian/confluence/plugins/content_report/js/tablesorter-date-parser.js' */
require(["ajs","jquery","confluence-sortable-tables/hooks"],function(d,c,a){a.onBeforeInit(function(a){a.addParser({id:"dateAttributeParser",is:function(a,e,b){return c(b).is(".content-report-table-macro .modified")},format:function(a,e,b,d){return c(b).attr("data-sortable-date")},type:"numeric"})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-content-report-plugin:resources', location = 'com/atlassian/confluence/plugins/content_report/soy/content-report-table.soy' */
// This file was automatically generated from content-report-table.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Plugins.ContentReport.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Plugins == 'undefined') { Confluence.Templates.Plugins = {}; }
if (typeof Confluence.Templates.Plugins.ContentReport == 'undefined') { Confluence.Templates.Plugins.ContentReport = {}; }


Confluence.Templates.Plugins.ContentReport.contentReportTable = function(opt_data, opt_ignored) {
  var output = '';
  var hasSocialColumn__soy3 = opt_data.showCommentsCount || opt_data.showLikesCount;
  if (opt_data.results.length == 0 && opt_data.blueprintKey) {
    output += '<div class="blueprint-blank-experience ' + soy.$$escapeHtml(opt_data.blueprintKey) + '"><div class="content"><h2>' + soy.$$escapeHtml(opt_data.blankTitle) + '</h2><p>' + soy.$$escapeHtml(opt_data.blankDescription) + '</p></div>' + ((opt_data.createButtonLabel) ? '<p><button class="create-from-template-button aui-button aui-button-primary" data-space-key="' + soy.$$escapeHtml(opt_data.dataSpaceKey) + '" data-content-blueprint-id="' + soy.$$escapeHtml(opt_data.contentBlueprintId) + '" href="' + soy.$$escapeHtml(opt_data.createContentUrl) + '" >' + soy.$$escapeHtml(opt_data.createButtonLabel) + '</button></p>' : '') + '</div>';
  } else {
    output += '<table class="aui content-report-table-macro' + ((hasSocialColumn__soy3) ? ' with-extra-columns' : '') + '"' + ((opt_data.analyticsKey) ? ' data-analytics-key="' + soy.$$escapeHtml(opt_data.analyticsKey) + '"' : '') + '><thead><tr><th>' + soy.$$escapeHtml('Title') + '</th><th>' + soy.$$escapeHtml('Creator') + '</th><th>' + soy.$$escapeHtml('Modified') + '</th></tr></thead><tbody>';
    var resultList43 = opt_data.results;
    var resultListLen43 = resultList43.length;
    if (resultListLen43 > 0) {
      for (var resultIndex43 = 0; resultIndex43 < resultListLen43; resultIndex43++) {
        var resultData43 = resultList43[resultIndex43];
        output += '<tr><td class="title"><a href="' + soy.$$escapeHtml(resultData43.urlPath) + '">' + soy.$$escapeHtml(resultData43.title) + '</a></td><td class="creator">' + Confluence.Templates.User.usernameLink({canView: opt_data.canViewProfiles, username: resultData43.creatorName, fullName: resultData43.creatorFullName, contextPath: opt_data.contextPath}) + '</td><td class="modified" data-sortable-date="' + soy.$$escapeHtml(resultData43.sortableDate) + '">' + soy.$$escapeHtml(resultData43.friendlyModificationDate) + '</td>' + ((hasSocialColumn__soy3) ? '<td class="social">' + ((opt_data.showCommentsCount && resultData43.commentCount != 0) ? '<span class="icon icon-comment"></span> <span class="count">' + soy.$$escapeHtml(resultData43.commentCount) + '</span>' : '') + ((opt_data.showLikesCount && resultData43.likeCount != 0) ? '<span class="icon icon-like"></span> <span class="count">' + soy.$$escapeHtml(resultData43.likeCount) + '</span>' : '') + '</td>' : '') + '</tr>';
      }
    } else {
      output += '<tr><td colspan="3">' + soy.$$escapeHtml('No content found.') + '</td></tr>';
    }
    output += '</tbody></table>' + ((opt_data.searchMoreResultsLinkUrl) ? '<div class="more-results"><a href="' + soy.$$escapeHtml("/wiki") + soy.$$escapeHtml(opt_data.searchMoreResultsLinkUrl) + '">' + soy.$$escapeHtml('Find more results') + '</a></div>' : '');
  }
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.Plugins.ContentReport.contentReportTable.soyTemplateName = 'Confluence.Templates.Plugins.ContentReport.contentReportTable';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-content-report-plugin:resources', location = 'com/atlassian/confluence/plugins/content_report/js/content-report-analytics.js' */
require(["ajs"],function(b){b.toInit(function(c){c(".content-report-table-macro").on("click",".title a",function(a){(a=c(a.delegateTarget).data("analytics-key"))&&b.trigger("analytics",{name:"content-report-table-macro.content-click."+a})})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-sortable-tables:sortable-table-loader', location = 'js/loader.js' */
require(["ajs","wrm"],function(a,b){a.toInit(function(){a.$("#main").find("table").length&&b.require("wrc!sortable-tables-resources")})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-highlight-actions:highlighting-experiment-resources', location = '/js/doctheme-utils.js' */
Confluence.DocThemeUtils=Confluence.DocThemeUtils||function(a){function d(){return c().length?!0:!1}function c(){b||(b=a("#splitter-content"));return b}var b;return{isDocTheme:d,appendAbsolutePositionedElement:function(b){var e=a(b);a(b).appendTo(d()?c():a("body"));return e},getMainContentScrollTop:function(){return d()?c().scrollTop():a(document).scrollTop()},getMainContentScrollLeft:function(){return d()?c().scrollLeft():a(document).scrollLeft()},getDocThemeContentElement:c}}(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-highlight-actions:highlighting-experiment-resources', location = '/js/scrolling-inline-dialog.js' */
Confluence.ScrollingInlineDialog=function(x,y,z,d){var g=Confluence.DocThemeUtils.getDocThemeContentElement(),h=Confluence.DocThemeUtils.isDocTheme();d=d||{};!d.container&&h&&(d.container=g);var A=function(e,b,d,f){var p="auto",u=-7,v=h?g.width():AJS.$(window).width(),a=b.target.position();a.top+=h?g.scrollTop():0;a.left+=h?g.scrollLeft():0;var l=b.target.outerWidth();var t=a.left+l/2,w=h?g.scrollTop()+AJS.$(window).height()-AJS.$("#footer").outerHeight():(window.pageYOffset||document.documentElement.scrollTop)+
AJS.$(window).height();l=a.top+b.target.outerHeight()+f.offsetY;var n=e.find(".arrow").outerWidth(),k=e.outerWidth();var c=b.target.outerWidth();f.centerOnHighlight?k>c?(b=a.left-(k-c)/2,c=t-b-n/2):(b=a.left+f.offsetX,c=(k-n)/2):(b=a.left+f.offsetX,c=k>c?t-b-n/2:(k-n)/2);c=0>c?0:c;var m=h?a.top-g.scrollTop():a.top-(window.pageYOffset||document.documentElement.scrollTop),q=f.maxHeight||0,r=e.height();m=m>Math.max(r,q);q=l+e.height()<w;e=!q&&m||f.onTop;f.onTop=e;m=v-(b+k+10);e&&(l=a.top-r-8,u=r);!1===
e&&!1===q&&(a=l+r-w,a=h?g.scrollTop()+a+AJS.$("#footer").outerHeight():(window.pageYOffset||document.documentElement.scrollTop)+a+20,(h?g:AJS.$("html, body")).stop().animate({scrollTop:a},500));f.isRelativeToMouse?0>m?(p=10,b="auto",c=d.x-(AJS.$(window).width()-f.width)):(b=d.x-20,c=d.x-b):0>m&&(p=10,b="auto",c=t-(v-p-k)-n/2);return{displayAbove:e,popupCss:{left:b,right:p,top:l},arrowCss:{position:"absolute",left:c,right:"auto",top:u}}};d.calculatePositions||(d.calculatePositions=A);return AJS.InlineDialog.call(this,
x,y,z,d)};
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-highlight-actions:highlighting-experiment-resources', location = '/js/highlight-panel.js' */
Confluence.HighlightAction=function(d){function n(e){var c={};c.pageId=e.pageId;c.selectedText=e.selectedText;c.index=e.index;c.numMatches=e.numMatches;c.lastFetchTime=d("meta[name\x3d'confluence-request-time']").attr("content");return c}var l={},m,p={MAINCONTENT_AND_COMMENT:function(e){return Confluence.HighlightAction.RangeHelper.isSelectionInsideContent(d(".wiki-content"),e)},MAINCONTENT_ONLY:function(e){m=m||d(".wiki-content").first();return Confluence.HighlightAction.RangeHelper.isSelectionInsideContent(m,
e)},COMMENT_ONLY:function(e){return Confluence.HighlightAction.RangeHelper.isSelectionInsideContent(d(".comment-content"),e)}};return{registerButtonHandler:function(e,c){l[e]=d.extend({onClick:function(){},shouldDisplay:p.MAINCONTENT_AND_COMMENT},c)},getButtonHandler:function(d){var c=l[d];c||(c=function(){AJS.logError("The button with key "+d+" doesn't have a registered handler")});return c},insertContentAtSelectionEnd:function(e){var c=Confluence.getContextPath()+"/rest/highlighting/1.0/insert-storage-fragment";
return d.ajax({type:"POST",contentType:"application/json",url:c,data:JSON.stringify(e)})},insertContentsInTableColumnCells:function(e){var c=Confluence.getContextPath()+"/rest/highlighting/1.0/insert-storage-column-table";return d.ajax({type:"POST",contentType:"application/json",url:c,data:JSON.stringify(e)})},createTableInsertionBean:function(d,c,g){g=n(g);g.tableColumnIndex=c;g.cellModifications=d;return g},createInsertionBean:function(d,c){c=n(c);c.xmlModification=d[0].xmlInsertion;return c},createXMLModificationBean:function(d,
c){c=n(c);c.xmlModification=d;return c},clearTextSelection:function(){window.getSelection?(window.getSelection().empty&&window.getSelection().empty(),window.getSelection().removeAllRanges&&window.getSelection().removeAllRanges()):window.document.selection&&window.document.selection.empty()},WORKING_AREA:p}}(AJS.$);
Confluence.HighlightAction.RangeHelper=function(d){function n(a){var c=Confluence.DocThemeUtils.getMainContentScrollTop(),k=Confluence.DocThemeUtils.getMainContentScrollLeft(),f=a.getClientRects();if(!f.length&&a.parentElement()){f=d(a.parentElement());var h=f.offset();f=[{top:h.top-(Confluence.DocThemeUtils.isDocTheme()?0:c),left:h.left-(Confluence.DocThemeUtils.isDocTheme()?0:k),bottom:h.top+f.height(),right:h.left+f.width()}]}f=e(a,f);h=function(a){if(Confluence.DocThemeUtils.isDocTheme()){var c=
Confluence.DocThemeUtils.getDocThemeContentElement().offset();a.left-=c.left;a.right-=c.left;a.top-=c.top;a.bottom-=c.top}return a};a=h(function(a,d){var b={};b.top=a.top;b.left=a.left+k;b.bottom=d.bottom;b.right=a.left>=d.right?a.right:d.right;b.right+=k;b.top+=c;b.bottom+=c;b.width=b.right-b.left;b.height=b.bottom-b.top;return b}(f.first,f.last));f=h(function(a){var b={};b.width=a.right-a.left;b.height=a.bottom-a.top;b.left=a.left+k;b.right=a.right+k;b.top=a.top+c;b.bottom=a.bottom+c;return b}(f.first));
Confluence.HighlightAction.debug&&(h=d("\x3cdiv\x3e").attr("id","highlight-actions-debug-helper"),Confluence.DocThemeUtils.appendAbsolutePositionedElement(h).css(d.extend({position:"absolute",outline:"1px solid red"},a)));return{first:f,average:a}}function l(a){return(void 0!=a.text?a.text:a.cloneContents().textContent).replace(/\u00a0/g," ")}function m(a){return a.cloneContents?d("\x3cdiv\x3e").append(a.cloneContents()).html():a.htmlText}function p(a){if(a.commonAncestorContainer)return a=a.commonAncestorContainer,
3==a.nodeType?a.parentNode:a;if(a.parentElement)return a.parentElement()}function e(a,b){var c={};c.first=b[0];c.last=b[b.length-1];"undefined"!==a.endOffset&&0==a.endOffset&&1<b.length&&(c.last=b[b.length-2]);return c}function c(a,b){var c=p(b);return function(){var b=!1;d.each(a,function(a,f){if(f===c||d.contains(f,c))return b=!0,!1});return b}()}function g(a){if(document.createRange)return a.text();range=document.body.createTextRange();range.moveToElementText(a.get(0));return range.text}function q(a,
b,c){var f=b.find('.user-mention, a[href^\x3d"/"], a[href^\x3d"#"], thead:hidden.tableFloatingHeader, div[data-macro-name\x3d"panel"] .panelHeader');b.find('.conf-macro[data-hasbody\x3d"false"], .jira-issue, .jira-issues').each(function(){-1<d(this).text().indexOf(a)&&(f=f.add(this))});if(0<f.length){var k=a.replace(/\S/g," "),e=new RegExp(a.replace(/[-\/\\^$*+?.()|[\]{}]/g,"\\$\x26"),"g");f.each(function(){var a=d(this).text();d(this).text(a.replace(e,k))});return g(b)}return c}return{getRangeOption:function(a){return{area:n(a),
text:l(a),html:m(a),containingElement:p(a),range:a}},getUserSelectionRange:function(){if(window.getSelection&&window.getSelection().isCollapsed||document.selection&&("None"==document.selection.type||""==document.selection.createRange().htmlText))return!1;if(window.getSelection){var a=window.getSelection();a=a.getRangeAt(a.rangeCount-1)}else document.selection&&(a=document.selection.createRange());if(/^\s*$/.test(l(a))){var b=m(a);if(!b||-1==b.toLowerCase().indexOf("\x3cimg "))return!1}return c(d(".wiki-content"),
a)?a:!1},getSelectionRects:n,getSelectionText:l,getSelectionHTML:m,getContainingElement:p,getFirstAndLastSelectionRect:e,isSelectionInsideContent:c,computeSearchTextObject:function(a,b){if(document.createRange){var c=document.createRange();c.setStart(a.get(0),0);c.setEnd(b.endContainer,b.originalEndOffset)}else c=document.body.createTextRange(),c.moveToElementText(a.get(0)),c.setEndPoint("EndToEnd",b);c=l(c);b=d.trim(l(b));var f=g(a);f=q(b,a.clone(),f);f=f.replace(/\u00a0/g," ");var e=0;for(a=[];-1<
(e=f.indexOf(b,e));)a.push(e),e+=1;c=c.replace(/\s*$/,"");return{pageId:AJS.Meta.get("page-id"),selectedText:b,index:d.inArray(c.length-b.length,a),numMatches:a.length}}}}(AJS.$);
AJS.toInit(function(d){function n(a){var b=m(),k=29*a.length,f=Confluence.HighlightPanel.Templates.panelContent({webItems:a}),h=!1,r=function(a){var b=!1;a.find("button").each(function(a){a=d(this);var c=a.attr("data-key");c=Confluence.HighlightAction.getButtonHandler(c).shouldDisplay(g);a.css("display",c?"inline-block":"none");b=b||c});b?a.find(".contents").width("auto"):c.hide()};c=Confluence.ScrollingInlineDialog(e,"selection-action-panel",function(a,b,e){h||(a.append(f),a.find(".aui-button").tooltip({gravity:"s"}),
a.parent().children().attr("unselectable","on").on("selectstart",!1),a.find("button").click(function(a){a=d(this).attr("data-key");a=Confluence.HighlightAction.getButtonHandler(a);c.hide();var b=Confluence.HighlightAction.RangeHelper.getRangeOption(g);""!==d.trim(b.text)&&(b.searchText=Confluence.HighlightAction.RangeHelper.computeSearchTextObject(p,g));a.onClick(b)}));e();h=!0;return!1},{centerOnHighlight:!0,onTop:!0,fadeTime:0,width:k,persistent:!0,initCallback:function(){Confluence.HighlightAction.Analytics.sendAnalyticsForOpeningHighlightOptionPanel();
r(this.popup);b.bindHideEvents();e.show()},hideCallback:function(){b.unbindHideEvents();e.hide()}})}function l(a){var b;d(document).on("mouseup",function(k){a.done(function(a){a&&0<a.length&&0===d(k.target).closest(".aui-inline-dialog").length&&setTimeout(function(){clearTimeout(b);var a=1E3;d(c[0]).is(":visible")&&(a=0);b=setTimeout(function(){g=Confluence.HighlightAction.RangeHelper.getUserSelectionRange();g.originalEndOffset=g.endOffset;if(g&&""!==d.trim(g.toString())){var a=Confluence.HighlightAction.RangeHelper.getSelectionRects(g),
b;if(b=a)Confluence.DocThemeUtils.appendAbsolutePositionedElement(e),b=!1,q&&a.first.top==q.first.top&&a.first.height==q.first.height&&a.first.left==q.first.left&&a.first.width==q.first.width||(e.css({top:a.first.top,height:a.first.height,left:a.first.left,width:a.first.width}),q=a,b=!0),b=b||!d(c[0]).is(":visible");b&&(d(c[0]).hide(),c.show())}else c.hide()},a)},0)})});a.done(function(){AJS.bind("quickedit.success",function(){c.hide()})})}function m(){var a=!1,b=function(){a||(d("body").bind("click.selection-action-panel.inline-dialog-check",
function(a){0===d(a.target).closest("#inline-dialog-selection-action-panel .contents").length&&(g||c.hide())}),a=!0)},e=function(a){27===a.keyCode&&c.hide()};return{bindHideEvents:function(){b();d(document).on("keydown",e)},unbindHideEvents:function(){a&&d("body").unbind("click.selection-action-panel.inline-dialog-check");a=!1;d(document).off("keydown",e)}}}var p=d(".wiki-content").first(),e=d("\x3cdiv\x3e").attr("id","action-dialog-target"),c,g,q;(function(){var a=Confluence.getContextPath()+"/rest/highlighting/1.0/panel-items",
b=AJS.Meta.get("page-id");void 0!=b&&(a=a+"?pageId\x3d"+b);a=d.get(a,function(a){a.length&&n(a)});l(a)})()});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-highlight-actions:highlighting-experiment-resources', location = '/js/quote-in-comment.js' */
AJS.toInit(function(c){function f(){if(Confluence.DocThemeUtils.isDocTheme()){var a=Confluence.DocThemeUtils.getDocThemeContentElement(),b=a.scrollTop()-c("#header").height()+c("#rte-toolbar").offset().top;a.scrollTop(b-40)}else b=c("#rte-toolbar").offset().top,c(document).scrollTop(b-40)}function d(a,b){a.execCommand("mceInsertContent",!1,"\x3cblockquote\x3e\x3cp\x3e"+b.html+"\x3c/p\x3e\x3c/blockquote\x3e\x3cp\x3e\x3cbr/\x3e\x3c/p\x3e")}function g(a){Confluence.HighlightAction.clearTextSelection();
setTimeout(function(){var b=AJS&&AJS.Rte&&AJS.Rte.getEditor&&AJS.Rte.getEditor();if(b)Confluence.HighlightAction.Analytics.sendAnalyticsForQuoteInComment(),f(),d(b,a);else{Confluence.HighlightAction.Analytics.sendAnalyticsForQuoteInComment(!0);var e=function(){d(AJS.Rte.getEditor(),a);AJS.unbind("quickedit.visible",e)};AJS.bind("quickedit.visible",e);h(c(a.containingElement).closest("div.comment"))}},0)}function h(a){0< !a.length?c(".quick-comment-prompt").click():a.find(".comment-actions .action-reply-comment").click()}
Confluence&&Confluence.HighlightAction&&Confluence.HighlightAction.registerButtonHandler("com.atlassian.confluence.plugins.confluence-highlight-actions:quote-comment",{onClick:g,shouldDisplay:Confluence.HighlightAction.WORKING_AREA.MAINCONTENT_AND_COMMENT})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-highlight-actions:highlighting-experiment-resources', location = '/js/highlight-analytics.js' */
Confluence.HighlightAction.Analytics=function(d){function a(a,c){AJS.trigger("analytics",{name:a,data:c})}return{sendAnalyticsForOpeningHighlightOptionPanel:function(){a("confluence.highlight.actions.open")},sendAnalyticsForQuoteInComment:function(b){b?a("confluence.quote.in.comment.insert"):a("confluence.quote.in.comment.append")}}}(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-highlight-actions:highlighting-experiment-resources', location = '/soy/templates.soy' */
// This file was automatically generated from templates.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.HighlightPanel.Templates.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.HighlightPanel == 'undefined') { Confluence.HighlightPanel = {}; }
if (typeof Confluence.HighlightPanel.Templates == 'undefined') { Confluence.HighlightPanel.Templates = {}; }


Confluence.HighlightPanel.Templates.panelContent = function(opt_data, opt_ignored) {
  var output = '';
  var webItemList3 = opt_data.webItems;
  var webItemListLen3 = webItemList3.length;
  for (var webItemIndex3 = 0; webItemIndex3 < webItemListLen3; webItemIndex3++) {
    var webItemData3 = webItemList3[webItemIndex3];
    output += '<button data-key="' + soy.$$escapeHtml(webItemData3.key) + '" class="aui-button aui-button-compact aui-button-subtle" title="' + soy.$$escapeHtml(webItemData3.label) + '"><span class="aui-icon aui-icon-small ' + soy.$$escapeHtml(webItemData3.styleClass) + '"></span></button>';
  }
  return output;
};
if (goog.DEBUG) {
  Confluence.HighlightPanel.Templates.panelContent.soyTemplateName = 'Confluence.HighlightPanel.Templates.panelContent';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.expand-macro:expand-macro-core', location = 'com/atlassian/confluence/plugins/expand/js/expand-macro-core.js' */
(function(){var g=function(a){this.$=a;this.createToggleFunction=function(c){var a=this.$;return function(b){if("undefined"==typeof c||c(b)){b=a(this);var e=a(".expand-icon",b),d=a(".expand-content",b.closest(".expand-container")).first();d.hasClass("expand-hidden")?(e.removeClass("aui-iconfont-chevron-right").addClass("aui-iconfont-chevron-down"),d.css("display","block"),d.animate({opacity:1}),b.attr("aria-expanded",!0),b="expand"):(e.removeClass("aui-iconfont-chevron-down").addClass("aui-iconfont-chevron-right"),
d.animate({opacity:0},{complete:function(){d.hide()}}),b.attr("aria-expanded",!1),b="collapse");d.toggleClass("expand-hidden");"expand"===b?AJS.trigger("confluence.expand-macro.expanded"):AJS.trigger("confluence.expand-macro.collapsed");AJS.trigger("analyticsEvent",{name:"confluence.expand-macro.expand-click",data:{userAction:b}})}}};this.getExpandElements=function(a){return this.$("div[id^\x3d'expander-control-'].expand-control",a)}};"undefined"===typeof Confluence&&(Confluence={});"undefined"===
typeof Confluence.Plugins&&(Confluence.Plugins={});Confluence.Plugins.ExpandMacro={bind:function(a,c,f,b){a=new g(a);c=a.getExpandElements(c);c.length&&c.bind(f,a.createToggleFunction(b))}}})();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.expand-macro:expand-macro-desktop-resources', location = 'com/atlassian/confluence/plugins/expand/js/expand-macro.js' */
AJS.toInit(function(a){Confluence.Plugins.ExpandMacro.bind(a,a("body"),"click keyup",function(a){return!("keyup"==a.type&&13!=a.keyCode)})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-business-blueprints:sharelinks-urlmacro-resources', location = 'com/atlassian/confluence/plugins/sharelinksurlmacro/soy/sharelinks-urlmacro-templates.soy' */
// This file was automatically generated from sharelinks-urlmacro-templates.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Blueprints.SharelinksUrlMacro.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Blueprints == 'undefined') { Confluence.Blueprints = {}; }
if (typeof Confluence.Blueprints.SharelinksUrlMacro == 'undefined') { Confluence.Blueprints.SharelinksUrlMacro = {}; }


Confluence.Blueprints.SharelinksUrlMacro.bookmarkletLink = function(opt_data, opt_ignored) {
  return '<a class="aui-button sharelinks-urlmacro-button" href="' + Confluence.Blueprints.SharelinksUrlMacro.bookmarkletScript(opt_data) + '"><span>' + soy.$$escapeHtml('Share on Confluence') + '</span></a>';
};
if (goog.DEBUG) {
  Confluence.Blueprints.SharelinksUrlMacro.bookmarkletLink.soyTemplateName = 'Confluence.Blueprints.SharelinksUrlMacro.bookmarkletLink';
}


Confluence.Blueprints.SharelinksUrlMacro.bookmarkletScript = function(opt_data, opt_ignored) {
  return 'javascript:(function(){var screenWidth=screen.width,screenHeight=screen.height,popupWidth=640,popupHeight=580,popupLeft=0,popupTop=0; if(screenWidth>popupWidth){popupLeft=Math.round((screenWidth/2)-(popupWidth/2));}if(screenHeight>popupHeight){popupTop=Math.round((screenHeight/2)-(popupHeight/2));}window.open(\'' + soy.$$filterNoAutoescape(opt_data.bookmarkletActionURL) + '?bookmarkedURL=\'+encodeURIComponent(window.location.href), \'\',\'left=\'+popupLeft+\',top=\'+popupTop+\',width=\'+popupWidth+\',height=\'+popupHeight+\',personalbar=0,toolbar=0,scrollbars=1,resizable=1\');}());';
};
if (goog.DEBUG) {
  Confluence.Blueprints.SharelinksUrlMacro.bookmarkletScript.soyTemplateName = 'Confluence.Blueprints.SharelinksUrlMacro.bookmarkletScript';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-business-blueprints:sharelinks-urlmacro-resources', location = 'com/atlassian/confluence/plugins/sharelinksurlmacro/js/sharelinks-urlmacro.js' */
AJS.toInit(function(a){a(".sharelinks-urlmacro-button").click(function(){alert("Drag this link to your toolbar");return!1})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:jsUri', location = '/includes/js/third-party/jsUri.js' */
(function(){function g(a){a=decodeURIComponent(a);return a=a.replace("+"," ")}function h(a){var c,b,f,e,d=[];if("undefined"===typeof a||null===a||""===a)return d;0===a.indexOf("?")&&(a=a.substring(1));c=a.toString().split(/[&;]/);for(a=0;a<c.length;a++)b=c[a],f=b.split("="),e=f[0],b=-1===b.indexOf("=")?null:null===f[1]?"":f[1],d.push([e,b]);return d}function d(a){var c=/^(?:(?![^:@]+:[^:@\/]*@)([^:\/?#.]+):)?(?:\/\/)?((?:(([^:@\/]*)(?::([^:@\/]*))?)?@)?([^:\/?#]*)(?::(\d*))?)(((\/(?:[^?#](?![^?#\/]*\.[^?#\/.]+(?:[?#]|$)))*\/?)?([^?#\/]*))(?:\?([^#]*))?(?:#(.*))?)/.exec(a||
""),b={};"source protocol authority userInfo user password host port relative path directory file query anchor".split(" ").forEach(function(a,e){b[a]=c[e]||""});this.uriParts=b;this.queryPairs=h(this.uriParts.query);this.hasAuthorityPrefixUserPref=null}Array.prototype.forEach||(Array.prototype.forEach=function(a,c){for(var b=0,f=this.length;b<f;++b)a.call(c||this,this[b],b,this)});"protocol userInfo host port path anchor".split(" ").forEach(function(a){d.prototype[a]=function(c){typeof c!=="undefined"&&
(this.uriParts[a]=c);return this.uriParts[a]}});d.prototype.hasAuthorityPrefix=function(a){if(typeof a!=="undefined")this.hasAuthorityPrefixUserPref=a;return this.hasAuthorityPrefixUserPref===null?this.uriParts.source.indexOf("//")!==-1:this.hasAuthorityPrefixUserPref};d.prototype.query=function(a){var c="",b;if(typeof a!=="undefined")this.queryPairs=h(a);for(a=0;a<this.queryPairs.length;a++){b=this.queryPairs[a];c.length>0&&(c=c+"&");c=b[1]===null?c+b[0]:c+b.join("=")}return c.length>0?"?"+c:c};
d.prototype.getQueryParamValue=function(a){var c,b;for(b=0;b<this.queryPairs.length;b++){c=this.queryPairs[b];if(g(a)===g(c[0]))return c[1]}};d.prototype.getQueryParamValues=function(a){var c=[],b,f;for(b=0;b<this.queryPairs.length;b++){f=this.queryPairs[b];g(a)===g(f[0])&&c.push(f[1])}return c};d.prototype.deleteQueryParam=function(a,c){var b=[],f,e,d,h;for(f=0;f<this.queryPairs.length;f++){e=this.queryPairs[f];d=g(e[0])===g(a);h=g(e[1])===g(c);(arguments.length===1&&!d||arguments.length===2&&!d&&
!h)&&b.push(e)}this.queryPairs=b;return this};d.prototype.addQueryParam=function(a,c,b){if(arguments.length===3&&b!==-1){b=Math.min(b,this.queryPairs.length);this.queryPairs.splice(b,0,[a,c])}else arguments.length>0&&this.queryPairs.push([a,c]);return this};d.prototype.replaceQueryParam=function(a,c,b){var d=-1,e,h;if(arguments.length===3){for(e=0;e<this.queryPairs.length;e++){h=this.queryPairs[e];if(g(h[0])===g(a)&&decodeURIComponent(h[1])===g(b)){d=e;break}}this.deleteQueryParam(a,b).addQueryParam(a,
c,d)}else{for(e=0;e<this.queryPairs.length;e++){h=this.queryPairs[e];if(g(h[0])===g(a)){d=e;break}}this.deleteQueryParam(a);this.addQueryParam(a,c,d)}return this};"protocol hasAuthorityPrefix userInfo host port path query anchor".split(" ").forEach(function(a){var c="set"+a.charAt(0).toUpperCase()+a.slice(1);d.prototype[c]=function(b){this[a](b);return this}});d.prototype.scheme=function(){var a="";if(this.protocol()){a=a+this.protocol();this.protocol().indexOf(":")!==this.protocol().length-1&&(a=
a+":");a=a+"//"}else this.hasAuthorityPrefix()&&this.host()&&(a=a+"//");return a};d.prototype.origin=function(){var a=this.scheme();if(this.userInfo()&&this.host()){a=a+this.userInfo();this.userInfo().indexOf("@")!==this.userInfo().length-1&&(a=a+"@")}if(this.host()){a=a+this.host();this.port()&&(a=a+(":"+this.port()))}return a};d.prototype.toString=function(){var a=this.origin();if(this.path())a=a+this.path();else if(this.host()&&(this.query().toString()||this.anchor()))a=a+"/";if(this.query().toString()){this.query().toString().indexOf("?")!==
0&&(a=a+"?");a=a+this.query().toString()}if(this.anchor()){this.anchor().indexOf("#")!==0&&(a=a+"#");a=a+this.anchor()}return a};d.prototype.clone=function(){return new d(this.toString())};define("confluence/jsUri",function(){return d})})(this);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:inline-tasks-resources', location = 'js/inline-tasks.js' */
(function(b){function g(a){var c;if(void 0===a.offsetX){var b=c=0,d=a.target;do{if(0!=d.scrollTop||0!=d.scrollLeft)var h=d;c+=d.offsetLeft;b+=d.offsetTop;d=d.offsetParent}while(d&&d!=d.offsetParent);c=a.pageX+(h?h.scrollLeft:0)-c;a=a.pageY+(h?h.scrollTop:0)-b}else c=a.offsetX,a=a.offsetY;return 3<=c&&14>=c&&3<=a&&14>=a}function k(a){var c="page";a.closest("table.tasks-report").length?c="report":a.closest("#task-container").length?c="mytasks":a.closest("ul.inline-task-list").length&&(c="task");return c}
function l(a,c){var f=a.attr("data-inline-task-id");a=a.find(c).first();return a.closest("li").attr("data-inline-task-id")===f?a:b()}function m(a){a.attr("aria-checked",a.hasClass("checked")?"true":"false")}var n=!1;b(window).bind("beforeunload",function(){n=!0});var e=[];b(document).delegate("ul.inline-task-list \x3e li[data-inline-task-id]",{click:function(a){if(a.currentTarget===a.target&&g(a)){var c=b(a.target);c.toggleClass("checked");m(c);var f=c.hasClass("checked")?"CHECKED":"UNCHECKED",d=
c.data("inline-task-id");a=c.closest("ul").attr("data-inline-tasks-content-id")||AJS.params.pageId;a=AJS.contextPath()+"/rest/inlinetasks/1/task/"+a+"/"+d+"/";c.prop("disabled",!0);c.closest("tr").attr("aria-disabled",!0);e.push(d);AJS.trigger("inline-tasks.status-update.start",{status:f,taskId:d,taskListQueue:e});b.ajax({type:"POST",url:a,data:b.toJSON({status:f,trigger:"VIEW_PAGE"}),dataType:"json",contentType:"application/json",timeout:3E4,error:function(a,b,e){n||"timeout"===b||(AJS.logError("Inline Task #"+
d+" was not persisted to "+f+" because of "+e+" (status: "+b+")"),b=require("confluence/message-controller"),c.toggleClass("checked"),m(c),b.showError(b.parseError(a,"Oops! Your task change couldn\u0027t be saved. \u003cbr/\u003eThere could be a few reasons for this."),b.Location.FLAG))},success:function(){var a=l(c,"time").attr("datetime"),b=new Date;var d=""+b.getFullYear();var e=""+(b.getMonth()+1);b=""+b.getDate();2>e.length&&(e="0"+e);2>b.length&&(b="0"+b);d=[d,e,b].join("-");a={dueDate:a,completionDate:d,mode:"view",assigneeUsername:l(c,".user-mention").attr("data-username"),
context:k(c)};"CHECKED"===f&&AJS.trigger("analyticsEvent",{name:"confluence-spaces.tasks.completed",data:a})}}).always(function(){c.prop("disabled",!1);c.closest("tr").attr("aria-disabled",!1);e.splice(e.indexOf(d),1);AJS.trigger("inline-tasks.status-update.complete",{status:f,taskId:d,taskListQueue:e})})}},mousemove:function(a){a.currentTarget===a.target&&(g(a)?b(a.target).addClass("hover"):b(a.target).removeClass("hover"))},mouseout:function(a){a.currentTarget===a.target&&b(a.target).removeClass("hover")},
mousedown:function(a){a.currentTarget===a.target&&g(a)&&b(a.target).addClass("task-active")},mouseup:function(a){a.currentTarget===a.target&&g(a)&&b(a.target).removeClass("task-active")}});b(document).tooltip({title:function(){return "This task is due in less than a week"},live:"ul.inline-task-list li:not(.checked) time.date-upcoming"});b(document).tooltip({title:function(){return "This task is overdue"},live:"ul.inline-task-list li:not(.checked) time.date-past"});b(document).tooltip({title:function(){return "A completion date wasn\u0027t recorded for this task"},
live:"span.emptycompletedate"});b(document).on("click","time",function(){var a=b(this);a={date:a.attr("datetime"),mode:"view",context:k(a)};AJS.trigger("analyticsEvent",{name:"confluence-spaces.date.clicked",data:a})});AJS.toInit(function(){b(".inline-task-list").attr("role","group");b(".inline-task-list [data-inline-task-id]:not(.checked)").attr("role","checkbox").attr("aria-checked","false");b(".inline-task-list [data-inline-task-id].checked").attr("role","checkbox").attr("aria-checked","true");
"READ_ONLY"===AJS.Meta.get("access-mode")&&(b("[data-inline-task-id]").addClass("disabled"),b("[data-inline-task-id]").prop("disabled",!0))})})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:inline-tasks-resources', location = 'js/inline-tasks-alert.js' */
Confluence.InlineTasks=Confluence.InlineTasks||{};
(function(c){var a=function(d){this.settings=c.extend({},a.DEFAULTS,d);this.template=Confluence.InlineTasks.Templates;c("#inline-tasks-notice").remove();var b=c(this.template.notice(this.settings));b.hide().appendTo("body");b.find(".notice-close").click(function(){b.hide()});this.$notice=b};a.DEFAULTS={textMessage:"Oops! Your task change couldn\u0027t be saved. \u003cbr/\u003eThere could be a few reasons for this.",className:"general-notice"};a.prototype.show=function(){this.$notice.show();return this};a.prototype.hide=function(){this.$notice.hide();
return this};Confluence.InlineTasks.Notice=a})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:inline-tasks-resources', location = 'templates/inline-tasks.soy' */
// This file was automatically generated from inline-tasks.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.InlineTasks.Templates.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.InlineTasks == 'undefined') { Confluence.InlineTasks = {}; }
if (typeof Confluence.InlineTasks.Templates == 'undefined') { Confluence.InlineTasks.Templates = {}; }


Confluence.InlineTasks.Templates.notice = function(opt_data, opt_ignored) {
  return '<div class="aui-message error' + ((opt_data.className) ? ' ' + soy.$$escapeHtml(opt_data.className) : '') + '" id="inline-tasks-notice">' + soy.$$filterNoAutoescape(opt_data.textMessage) + '&nbsp;&nbsp;<a href="#" class="notice-close">' + soy.$$escapeHtml('Dismiss') + '</a></div>';
};
if (goog.DEBUG) {
  Confluence.InlineTasks.Templates.notice.soyTemplateName = 'Confluence.InlineTasks.Templates.notice';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:inline-tasks-resources', location = 'js/inline-tasks-focus.js' */
AJS.$(document).ready(function(a){if(a=(new (require("confluence/jsUri"))(window.location.href)).getQueryParamValue("focusedTaskId"))a=$("li[data-inline-task-id\x3d"+a+"]"),a.length&&(a.addClass("focused"),window.scrollTo(a.offset().left,a.offset().top-$(window).height()/2))});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:sortable-table-server-side', location = 'js/jquery.tablesorter.serveronly.js' */
(function(d){d.extend({tablesorterServerOnly:new function(){this.defaults={classNameDisableSorting:"aui-table-column-unsortable",classNameHeaderDesc:"tablesorter-headerDesc",classNameHeaderAsc:"tablesorter-headerAsc",reverseSort:!1,sortColumn:"",onInit:function(){},onSort:function(){},events:{clickHeader:"click.sortServerOnly",refreshHeader:"refreshHeader.sortServerOnly",simulateClickHeader:"simulateClickHeader.sortServerOnly"}};var e={updateCurrentHeaderSort:function(b,a){e.resetHeadersSort(b,a);
a.$headers.each(function(){var b=d(this),g=b.attr("data-column-name"),e=a.reverseSort;g===a.sortColumn&&(e?b.addClass(a.classNameHeaderDesc):b.addClass(a.classNameHeaderAsc))})},resetHeadersSort:function(b,a){a.$headers.removeClass(a.classNameHeaderDesc).removeClass(a.classNameHeaderAsc)},prepareHTMLHeader:function(b,a){a.$headers.not("."+a.classNameDisableSorting).attr("unselectable","on").bind("selectstart",!1).addClass("tablesorter-header").wrapInner("\x3cspan class\x3d'aui-table-header-content'/\x3e")},
bindEvents:function(b,a){var c=d(b);c.on(a.events.refreshHeader,function(){e.updateCurrentHeaderSort(b,a)});c.on(a.events.simulateClickHeader,function(b,d,c){a.reverseSort=c;a.sortColumn=d})}},f=function(b,a){var c=d(b);a.$table=c;a.$headers=c.find("thead th");a.$tbodies=c.find("tbody");e.prepareHTMLHeader(b,a);e.updateCurrentHeaderSort(b,a);"function"===typeof a.onInit&&a.onInit.apply(b,[a]);a.$headers.on(a.events.clickHeader,function(){var c=d(this);if(c.hasClass(a.classNameDisableSorting))return!1;
c=c.attr("data-column-name");a.reverseSort=c===a.sortColumn?!a.reverseSort:!1;a.sortColumn=c;"function"===typeof a.onSort&&a.onSort.apply(b,[a]);return!1});e.bindEvents(b,a)};this.construct=function(b){return this.each(function(){var a=this,c=d.extend(!0,{},d.tablesorterServerOnly.defaults,b);this.config&&a.hasInitialized&&d.tablesorter?d.tablesorter.destroy(a,!1,function(){f(a,c)}):f(a,c)})}}});d.fn.extend({tablesorterServerOnly:d.tablesorterServerOnly.construct})})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:sortable-table-server-side', location = 'js/tasks-table-sortable.js' */
(function(e){var d=function(a){this.ajaxUrl=a.ajaxUrl;this.restUrlPagination=a.restUrlPagination;this.$wrapper=a.$wrapper;this.table=a.table;this.$table=e(this.table);this.analyticEventKey=a.analyticEventKey;this.sortColumnDefault=a.sortColumnDefault||"duedate";this.sortReverseSortDefault=a.sortReverseSortDefault||!1;this.reportParametersDefault=a.reportParametersDefault;this.pageIndex=a.pageIndex||0;this.pageSize=a.pageSize||10;this.pageTotal=a.pageTotal||0;this.pageLimit=a.pageLimit||7;this.adaptive=
a.adaptive;this.columns=a.columns;this.templates=a.templates;this.onRenderEmptyTable=a.onRenderEmptyTable;this.onBusySorting=a.onBusySorting};d.getColumnNameFromSortBy=function(a){var b={"due date":"duedate","page title":"location",assignee:"assignee"};return b[a]?b[a]:"duedate"};d.getSortByFromColumnName=function(a){var b={duedate:"due date",location:"page title"};return b[a]?b[a]:a};d.prototype.updateOptions=function(a){e.extend(this,a);this.$table=e(this.table)};d.prototype.getCurrentPageIndex=
function(){var a=this.$wrapper.find(".macro-auto-pagination").last();a=parseInt(a.attr("data-initial-page-index"),10);return isNaN(a)?0:a};d.prototype.renderPagination=function(a,b){var c=this;a||(a=c.$table);b||(b=c.reportParametersDefault);this.$wrapper.find(".macro-auto-pagination").remove();1<c.pageTotal&&f.UI.Components.Pagination.build({scope:a,pageSize:c.pageSize,totalPages:c.pageTotal,pageLimit:c.pageLimit,path:c.restUrlPagination,adaptive:c.adaptive,currentPage:c.pageIndex,data:{reportParameters:JSON.stringify(b)},
success:function(a,b){a=c.templates.tasksReportLine({task:a,columns:c.columns});b.append(a)}})};d.prototype.toggleBusyState=function(a){this.$wrapper.attr("data-loading",a);a?this.$wrapper.find(".task-blanket").show():this.$wrapper.find(".task-blanket").hide();"function"===typeof this.onBusySorting&&this.onBusySorting.apply(this,[a])};d.prototype.renderTable=function(a){var b=this;a=_.map(a,function(a){return b.templates.tasksReportLine({task:a,columns:b.columns})}).join("");b.$table.find("tbody").html(a);
f.Binder.userHover()};d.prototype._triggerAnalyticsSorting=function(){AJS.trigger("analyticsEvent",{name:this.analyticEventKey,data:{column:this.sortColumn,direction:this.reverseSort?"desc":"asc"}})};d.prototype._buildAjaxData=function(a){return{url:this.ajaxUrl,cache:!1,dataType:"json",data:{pageIndex:this.pageIndex,pageSize:this.pageSize,reportParameters:JSON.stringify(a)}}};d.prototype.init=function(){var a=this;a.sortColumn=a.sortColumnDefault;a.reverseSort=a.sortReverseSortDefault;this.$table.tablesorterServerOnly({sortColumn:a.sortColumn,
reverseSort:a.reverseSort,onInit:function(){e(this).addClass("aui-table-sortable")},onSort:function(b){var c=e(this);a.pageIndex=a.getCurrentPageIndex();a.sortColumn=b.sortColumn;a.reverseSort=b.reverseSort;a.toggleBusyState(!0);var g=e.extend({},a.reportParametersDefault,{sortColumn:d.getSortByFromColumnName(a.sortColumn),reverseSort:a.reverseSort});b=a._buildAjaxData(g);e.ajax(b).done(function(b){a.pageIndex=a.getCurrentPageIndex();a.pageTotal=b.totalPages;0===a.pageIndex&&0===a.pageTotal?"function"===
typeof a.onRenderEmptyTable&&(a.$wrapper.find(".macro-auto-pagination").remove(),c.remove(),a.onRenderEmptyTable.apply(a)):(a.renderTable(b.detailLines),a.renderPagination(null,g),c.trigger("refreshHeader.sortServerOnly"),a._triggerAnalyticsSorting())}).fail(function(){(new f.InlineTasks.Notice({textMessage:"We can\u0027t sort your tasks right now. Refresh the page to try again.",className:"forbidden-notice"})).show()}).always(function(){AJS.trigger("ic-jim-async-supported");a.toggleBusyState(!1)})}})};var f=
window.Confluence||{};f.InlineTasks=f.InlineTasks||{};f.InlineTasks.TasksTableSortable=d})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-ui-components:common', location = '/js/internal/namespace-legacy.js' */
var Confluence=require("confluence/legacy");Confluence.UI||(Confluence.UI={});Confluence.UI.Components||(Confluence.UI.Components={});Confluence.UI.Components.BlankPlaceholderBox||(Confluence.UI.Components.BlankPlaceholderBox={});Confluence.UI.Components.CQL||(Confluence.UI.Components.CQL={});Confluence.UI.Components.DatePicker||(Confluence.UI.Components.DatePicker={});Confluence.UI.Components.LabelPicker||(Confluence.UI.Components.LabelPicker={});
Confluence.UI.Components.Pagination||(Confluence.UI.Components.Pagination={});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-ui-components:common', location = 'js/internal/soy-templates.js' */
define("confluence-ui-components/js/internal/soy-templates",[],function(){return{BlankPlaceholderBox:function(){return Confluence.UI.Components.BlankPlaceholderBox.Templates},DatePicker:function(){return Confluence.UI.Components.DatePicker.templates},LabelPicker:function(){return Confluence.UI.Components.LabelPicker.templates},Pagination:function(){return Confluence.UI.Components.Pagination.Templates},Components:function(){return Confluence.UI.Components.templates},UserGroupSelect2:function(){return Confluence.UI.Components.UserGroupSelect2}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-ui-components:pagination', location = '/js/pagination.js' */
define("confluence-ui-components/js/pagination",["ajs","confluence/hover-user","confluence-ui-components/js/internal/soy-templates","jquery","underscore"],function(m,q,r,k,n){var l=function(a,b,c){var d=a.find("a").attr("aria-disabled",c);c?d.attr("disabled","disabled"):(d.removeAttr("disabled"),c=k(".aui-nav-selected",a).data("index")+1,a.find(".aui-nav-next \x3e a").attr("aria-disabled",c===b),a.find(".aui-nav-previous \x3e a").attr("aria-disabled",1===c))},t={scope:null,success:null,data:null,
path:"",url:"#",pageLimit:7,currentPage:0,adaptive:!1,totalPages:0,pageSize:0},p=function(a){var b=a.totalPages,c=a.currentPage,d=a.pageLimit,f=a.adaptive;d?(f&&(b=c+d),f=d):f=b;var e=Math.floor(d/2);d=!d||b<=d||0>c-e?0:c+e>b-1?b-d:c-e;return r.Pagination().paginationFooter({currentPage:c||0,startPage:d,itemsToRender:f,totalPages:b,pageSize:a.pageSize,url:a.url||"#"})};return{build:function(a){var b=n.extend({},t,a);"function"!==typeof b.success&&(b.success=function(){});a=p(b);b.scope.after(a);var c=
b.scope.next(),d=c.data("initial-page-index");c.on("click","a",function(a){var e=k(this),h=e.parents("ol").prev();"floating-scrollbar"===h.prop("id")&&(h=h.prev());var f=h.is("table")?h:h.find("table"),g=f.data("pageIndex")||d;e=e.parent("li");e.hasClass("aui-nav-selected")||e.find("\x3e a[aria-disabled\x3dtrue]").length||(e.hasClass("aui-nav-next")?g++:e.hasClass("aui-nav-previous")?g--:g=e.data("index"),l(c,b.totalPages,!0),e=k.extend({},{pageSize:b.pageSize,pageIndex:g},b.data),k.getJSON(m.contextPath()+
b.path,e).done(function(a){g=a.currentPage;var d=f.find("tbody");d.find("tr").remove();f.data("pageIndex",g);n.each(a.detailLines,function(a){b.success(a,d)});m.trigger("ui.components.pagination.updated",[a,b]);a=k.extend({},b,{totalPages:a.totalPages,adaptive:a.adaptive,currentPage:g});c.html(p(a));l(c,a.totalPages,!1);q()}).fail(function(){l(c,b.totalPages,!1)}),a.preventDefault())})}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-ui-components:pagination', location = '/js/internal/pagination-legacy.js' */
var _=require("underscore");window.Confluence.UI.Components.Pagination=_.extend(window.Confluence.UI.Components.Pagination,require("confluence-ui-components/js/pagination"));require("ajs").deprecate.prop(window.Confluence.UI.Components.Pagination,"build",{sinceVersion:"2.1.1",extraInfo:"Use require('confluence-ui-components/js/pagination')"});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-ui-components:pagination', location = '/soy/pagination.soy' */
// This file was automatically generated from pagination.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.UI.Components.Pagination.Templates.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.UI == 'undefined') { Confluence.UI = {}; }
if (typeof Confluence.UI.Components == 'undefined') { Confluence.UI.Components = {}; }
if (typeof Confluence.UI.Components.Pagination == 'undefined') { Confluence.UI.Components.Pagination = {}; }
if (typeof Confluence.UI.Components.Pagination.Templates == 'undefined') { Confluence.UI.Components.Pagination.Templates = {}; }


Confluence.UI.Components.Pagination.Templates.paginationFooter = function(opt_data, opt_ignored) {
  var output = '<ol class="aui-nav aui-nav-pagination macro-auto-pagination" data-page-size="' + soy.$$escapeHtml(opt_data.pageSize) + '" data-initial-page-index="' + soy.$$escapeHtml(opt_data.currentPage) + '"><li class="aui-nav-previous"><a ' + ((opt_data.currentPage == 0) ? 'aria-disabled="true"' : 'href="#"') + '>' + soy.$$escapeHtml('Prev') + '</a></li>';
  var startIdx__soy16 = opt_data.startPage + 1;
  var currentIdx__soy17 = opt_data.currentPage + 1;
  var endIdx__soy18 = startIdx__soy16 + opt_data.itemsToRender;
  var totalRange__soy19 = opt_data.totalPages + 1;
  var indexInit20 = startIdx__soy16;
  var indexLimit20 = endIdx__soy18 < totalRange__soy19 ? endIdx__soy18 : totalRange__soy19;
  for (var index20 = indexInit20; index20 < indexLimit20; index20++) {
    output += (index20 == currentIdx__soy17) ? '<li class="aui-nav-selected" data-index="' + soy.$$escapeHtml(index20 - 1) + '">' + soy.$$escapeHtml(index20) + '</li>' : '<li data-index="' + soy.$$escapeHtml(index20 - 1) + '"><a href="#">' + soy.$$escapeHtml(index20) + '</a></li>';
  }
  output += '<li class="aui-nav-next"><a ' + ((currentIdx__soy17 == opt_data.totalPages) ? 'aria-disabled="true"' : 'href="' + soy.$$escapeHtml(opt_data.url) + '"') + '>' + soy.$$escapeHtml('Next') + '</a></li></ol>';
  return output;
};
if (goog.DEBUG) {
  Confluence.UI.Components.Pagination.Templates.paginationFooter.soyTemplateName = 'Confluence.UI.Components.Pagination.Templates.paginationFooter';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-ui-components:blank-placeholder-box', location = 'soy/blank-placeholder-box.soy' */
// This file was automatically generated from blank-placeholder-box.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.UI.Components.BlankPlaceholderBox.Templates.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.UI == 'undefined') { Confluence.UI = {}; }
if (typeof Confluence.UI.Components == 'undefined') { Confluence.UI.Components = {}; }
if (typeof Confluence.UI.Components.BlankPlaceholderBox == 'undefined') { Confluence.UI.Components.BlankPlaceholderBox = {}; }
if (typeof Confluence.UI.Components.BlankPlaceholderBox.Templates == 'undefined') { Confluence.UI.Components.BlankPlaceholderBox.Templates = {}; }


Confluence.UI.Components.BlankPlaceholderBox.Templates.blankBox = function(opt_data, opt_ignored) {
  return '<div class="blank-placeholder-box ' + ((opt_data.customClass) ? soy.$$escapeHtml(opt_data.customClass) : '') + '"><div class="content"><h2>' + soy.$$escapeHtml(opt_data.blankTitle) + '</h2><p>' + soy.$$escapeHtml(opt_data.blankDescription) + '</p></div>' + ((opt_data.customHtml) ? soy.$$filterNoAutoescape(opt_data.customHtml) : '') + '</div>';
};
if (goog.DEBUG) {
  Confluence.UI.Components.BlankPlaceholderBox.Templates.blankBox.soyTemplateName = 'Confluence.UI.Components.BlankPlaceholderBox.Templates.blankBox';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.component.sortable-table', location = 'aui.chunk.181d293799bad4927095--bd78e9093cb91b429bfe.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.component.sortable-table"],{AqzS:function(e,t){
/**!
 * TableSorter 2.17.7 - Client-side table sorting with ease!
 * @requires jQuery v1.2.6+
 *
 * Copyright (c) 2007 Christian Bach
 * Examples and docs at: http://tablesorter.com
 * Dual licensed under the MIT and GPL licenses:
 * http://www.opensource.org/licenses/mit-license.php
 * http://www.gnu.org/licenses/gpl.html
 *
 * @type jQuery
 * @name tablesorter
 * @cat Plugins/Tablesorter
 * @author Christian Bach/christian.bach@polyester.se
 * @contributor Rob Garrison/https://github.com/Mottie/tablesorter
 */
!function(e){"use strict";e.extend({tablesorter:new function(){var t=this;function r(){var e=arguments[0],t=arguments.length>1?Array.prototype.slice.call(arguments):e;"undefined"!=typeof console&&void 0!==console.log?console[/error/i.test(e)?"error":/warn/i.test(e)?"warn":"log"](t):alert(t)}function s(e,t){r(e+" ("+((new Date).getTime()-t.getTime())+"ms)")}function a(e){for(var t in e)return!1;return!0}function n(r,s,a){if(!s)return"";var n,o=r.config,i=o.textExtraction||"",d="";return d="basic"===i?e(s).attr(o.textAttribute)||s.textContent||s.innerText||e(s).text()||"":"function"==typeof i?i(s,r,a):"function"==typeof(n=t.getColumnData(r,i,a))?n(s,r,a):s.textContent||s.innerText||e(s).text()||"",e.trim(d)}function o(e,s,a,o){for(var i,d=t.parsers.length,c=!1,l="",u=!0;""===l&&u;)s[++a]?(l=n(e,c=s[a].cells[o],o),e.config.debug&&r("Checking if value was empty on row "+a+", column: "+o+': "'+l+'"')):u=!1;for(;--d>=0;)if((i=t.parsers[d])&&"text"!==i.id&&i.is&&i.is(l,e,c))return i;return t.getParserById("text")}function i(e){var a,n,i,d,c,l,u,p,f,g,h=e.config,m=h.$tbodies=h.$table.children("tbody:not(."+h.cssInfoBlock+")"),b=0,y="",w=m.length;if(0===w)return h.debug?r("Warning: *Empty table!* Not building a parser cache"):"";for(h.debug&&(g=new Date,r("Detecting parsers for each column")),n={extractors:[],parsers:[]};b<w;){if((a=m[b].rows)[b])for(i=h.columns,d=0;d<i;d++)c=h.$headers.filter('[data-column="'+d+'"]:last'),l=t.getColumnData(e,h.headers,d),f=t.getParserById(t.getData(c,l,"extractor")),p=t.getParserById(t.getData(c,l,"sorter")),u="false"===t.getData(c,l,"parser"),h.empties[d]=t.getData(c,l,"empty")||h.emptyTo||(h.emptyToBottom?"bottom":"top"),h.strings[d]=t.getData(c,l,"string")||h.stringTo||"max",u&&(p=t.getParserById("no-parser")),f||(f=!1),p||(p=o(e,a,-1,d)),h.debug&&(y+="column:"+d+"; extractor:"+f.id+"; parser:"+p.id+"; string:"+h.strings[d]+"; empty: "+h.empties[d]+"\n"),n.parsers[d]=p,n.extractors[d]=f;b+=n.parsers.length?w:1}h.debug&&(r(y||"No parsers detected"),s("Completed detecting parsers",g)),h.parsers=n.parsers,h.extractors=n.extractors}function d(a){var o,i,d,c,l,u,p,f,g,h,m,b,y,w=a.config,v=w.$table.children("tbody"),x=w.extractors,C=w.parsers;if(w.cache={},w.totalRows=0,!C)return w.debug?r("Warning: *Empty table!* Not building a cache"):"";for(w.debug&&(h=new Date),w.showProcessing&&t.isProcessing(a,!0),p=0;p<v.length;p++)if(y=[],o=w.cache[p]={normalized:[]},!v.eq(p).hasClass(w.cssInfoBlock)){for(m=v[p]&&v[p].rows.length||0,l=0;l<m;++l)if(b={child:[]},f=e(v[p].rows[l]),[new Array(w.columns)],g=[],f.hasClass(w.cssChildRow)&&0!==l)i=o.normalized.length-1,o.normalized[i][w.columns].$row=o.normalized[i][w.columns].$row.add(f),f.prev().hasClass(w.cssChildRow)||f.prev().addClass(t.css.cssHasChild),b.child[i]=e.trim(f[0].textContent||f[0].innerText||f.text()||"");else{for(b.$row=f,b.order=l,u=0;u<w.columns;++u)void 0!==C[u]?(i=n(a,f[0].cells[u],u),d=void 0===x[u].id?i:x[u].format(i,a,f[0].cells[u],u),c="no-parser"===C[u].id?"":C[u].format(d,a,f[0].cells[u],u),g.push(w.ignoreCase&&"string"==typeof c?c.toLowerCase():c),"numeric"===(C[u].type||"").toLowerCase()&&(y[u]=Math.max(Math.abs(c)||0,y[u]||0))):w.debug&&r("No parser found for cell:",f[0].cells[u],"does it have a header?");g[w.columns]=b,o.normalized.push(g)}o.colMax=y,w.totalRows+=o.normalized.length}w.showProcessing&&t.isProcessing(a),w.debug&&s("Building cache for "+m+" rows",h)}function c(r,n){var o,i,d,c,l,u,p,f=r.config,g=f.widgetOptions,h=r.tBodies,m=[],b=f.cache;if(a(b))return f.appender?f.appender(r,m):r.isUpdating?f.$table.trigger("updateComplete",r):"";for(f.debug&&(p=new Date),u=0;u<h.length;u++)if((d=e(h[u])).length&&!d.hasClass(f.cssInfoBlock)){for(c=t.processTbody(r,d,!0),i=(o=b[u].normalized).length,l=0;l<i;l++)m.push(o[l][f.columns].$row),f.appender&&(!f.pager||f.pager.removeRows&&g.pager_removeRows||f.pager.ajax)||c.append(o[l][f.columns].$row);t.processTbody(r,c,!1)}f.appender&&f.appender(r,m),f.debug&&s("Rebuilt table",p),n||f.appender||t.applyWidget(r),r.isUpdating&&f.$table.trigger("updateComplete",r)}function l(e){return/^d/i.test(e)||1===e}function u(a){var n,o,i,d,c,u,p,g=a.config;g.headerList=[],g.headerContent=[],g.debug&&(p=new Date),g.columns=t.computeColumnIndex(g.$table.children("thead, tfoot").children("tr")),d=g.cssIcon?'<i class="'+(g.cssIcon===t.css.icon?t.css.icon:g.cssIcon+" "+t.css.icon)+'"></i>':"",g.$headers=e(a).find(g.selectorHeaders).each((function(r){o=e(this),n=t.getColumnData(a,g.headers,r,!0),g.headerContent[r]=e(this).html(),c=g.headerTemplate.replace(/\{content\}/g,e(this).html()).replace(/\{icon\}/g,d),g.onRenderTemplate&&(i=g.onRenderTemplate.apply(o,[r,c]))&&"string"==typeof i&&(c=i),e(this).html('<div class="'+t.css.headerIn+'">'+c+"</div>"),g.onRenderHeader&&g.onRenderHeader.apply(o,[r]),this.column=parseInt(e(this).attr("data-column"),10),this.order=l(t.getData(o,n,"sortInitialOrder")||g.sortInitialOrder)?[1,0,2]:[0,1,2],this.count=-1,this.lockedOrder=!1,void 0!==(u=t.getData(o,n,"lockedOrder")||!1)&&!1!==u&&(this.order=this.lockedOrder=l(u)?[1,1,1]:[0,0,0]),o.addClass(t.css.header+" "+g.cssHeader),g.headerList[r]=this,o.parent().addClass(t.css.headerRow+" "+g.cssHeaderRow).attr("role","row"),g.tabIndex&&o.attr("tabindex",0)})).attr({scope:"col",role:"columnheader"}),f(a),g.debug&&(s("Built headers:",p),r(g.$headers))}function p(e,t,r){var s=e.config;s.$table.find(s.selectorRemove).remove(),i(e),d(e),w(s.$table,t,r)}function f(r){var s,a,n,o=r.config;o.$headers.each((function(i,d){a=e(d),n=t.getColumnData(r,o.headers,i,!0),s="false"===t.getData(d,n,"sorter")||"false"===t.getData(d,n,"parser"),d.sortDisabled=s,a[s?"addClass":"removeClass"]("sorter-false").attr("aria-disabled",""+s),r.id&&(s?a.removeAttr("aria-controls"):a.attr("aria-controls",r.id))}))}function g(r){var s,a,n,o=r.config,i=o.sortList,d=i.length,c=t.css.sortNone+" "+o.cssNone,l=[t.css.sortAsc+" "+o.cssAsc,t.css.sortDesc+" "+o.cssDesc],u=["ascending","descending"],p=e(r).find("tfoot tr").children().add(o.$extraHeaders).removeClass(l.join(" "));for(o.$headers.removeClass(l.join(" ")).addClass(c).attr("aria-sort","none"),a=0;a<d;a++)if(2!==i[a][1]&&(s=o.$headers.not(".sorter-false").filter('[data-column="'+i[a][0]+'"]'+(1===d?":last":""))).length){for(n=0;n<s.length;n++)s[n].sortDisabled||s.eq(n).removeClass(c).addClass(l[i[a][1]]).attr("aria-sort",u[i[a][1]]);p.length&&p.filter('[data-column="'+i[a][0]+'"]').removeClass(c).addClass(l[i[a][1]])}o.$headers.not(".sorter-false").each((function(){var r=e(this),s=this.order[(this.count+1)%(o.sortReset?3:2)],a=r.text()+": "+t.language[r.hasClass(t.css.sortAsc)?"sortAsc":r.hasClass(t.css.sortDesc)?"sortDesc":"sortNone"]+t.language[0===s?"nextAsc":1===s?"nextDesc":"nextNone"];r.attr("aria-label",a)}))}function h(e,t){return e&&e[t]&&e[t].type||""}function m(r,s,a){if(r.isUpdating)return setTimeout((function(){m(r,s,a)}),50);var n,o,i,d,l,u=r.config,p=!a[u.sortMultiSortKey],f=u.$table;if(f.trigger("sortStart",r),s.count=a[u.sortResetKey]?2:(s.count+1)%(u.sortReset?3:2),u.sortRestart&&(o=s,u.$headers.each((function(){this===o||!p&&e(this).is("."+t.css.sortDesc+",."+t.css.sortAsc)||(this.count=-1)}))),o=s.column,p){if(u.sortList=[],null!==u.sortForce)for(n=u.sortForce,i=0;i<n.length;i++)n[i][0]!==o&&u.sortList.push(n[i]);if((d=s.order[s.count])<2&&(u.sortList.push([o,d]),s.colSpan>1))for(i=1;i<s.colSpan;i++)u.sortList.push([o+i,d])}else{if(u.sortAppend&&u.sortList.length>1)for(i=0;i<u.sortAppend.length;i++)(l=t.isValueInArray(u.sortAppend[i][0],u.sortList))>=0&&u.sortList.splice(l,1);if(t.isValueInArray(o,u.sortList)>=0)for(i=0;i<u.sortList.length;i++)l=u.sortList[i],d=u.$headers.filter('[data-column="'+l[0]+'"]:last')[0],l[0]===o&&(l[1]=d.order[s.count],2===l[1]&&(u.sortList.splice(i,1),d.count=-1));else if((d=s.order[s.count])<2&&(u.sortList.push([o,d]),s.colSpan>1))for(i=1;i<s.colSpan;i++)u.sortList.push([o+i,d])}if(null!==u.sortAppend)for(n=u.sortAppend,i=0;i<n.length;i++)n[i][0]!==o&&u.sortList.push(n[i]);f.trigger("sortBegin",r),setTimeout((function(){g(r),b(r),c(r),f.trigger("sortEnd",r)}),1)}function b(e){var r,n,o,i,d,c,l,u,p,f,g=0,m=e.config,b=m.textSorter||"",y=m.sortList,w=y.length,v=e.tBodies.length;if(!m.serverSideSorting&&!a(m.cache)){for(m.debug&&(d=new Date),n=0;n<v;n++)c=m.cache[n].colMax,m.cache[n].normalized.sort((function(s,a){for(r=0;r<w;r++){if(i=y[r][0],l=y[r][1],g=0===l,m.sortStable&&s[i]===a[i]&&1===w)return s[m.columns].order-a[m.columns].order;if((o=/n/i.test(h(m.parsers,i)))&&m.strings[i]?(o="boolean"==typeof m.string[m.strings[i]]?(g?1:-1)*(m.string[m.strings[i]]?-1:1):m.strings[i]&&m.string[m.strings[i]]||0,u=m.numberSorter?m.numberSorter(s[i],a[i],g,c[i],e):t["sortNumeric"+(g?"Asc":"Desc")](s[i],a[i],o,c[i],i,e)):(p=g?s:a,f=g?a:s,u="function"==typeof b?b(p[i],f[i],g,i,e):"object"==typeof b&&b.hasOwnProperty(i)?b[i](p[i],f[i],g,i,e):t["sortNatural"+(g?"Asc":"Desc")](s[i],a[i],i,e,m)),u)return u}return s[m.columns].order-a[m.columns].order}));m.debug&&s("Sorting on "+y.toString()+" and dir "+l+" time",d)}}function y(t,r){t[0].isUpdating&&t.trigger("updateComplete"),e.isFunction(r)&&r(t[0])}function w(e,r,s){var a=e[0].config.sortList;!1!==r&&!e[0].isProcessing&&a.length?e.trigger("sorton",[a,function(){y(e,s)},!0]):(y(e,s),t.applyWidget(e[0],!1))}function v(r){var s=r.config,o=s.$table;o.unbind("sortReset update updateRows updateCell updateAll addRows updateComplete sorton appendCache updateCache applyWidgetId applyWidgets refreshWidgets destroy mouseup mouseleave ".split(" ").join(s.namespace+" ")).bind("sortReset"+s.namespace,(function(t,a){t.stopPropagation(),s.sortList=[],g(r),b(r),c(r),e.isFunction(a)&&a(r)})).bind("updateAll"+s.namespace,(function(e,a,n){e.stopPropagation(),r.isUpdating=!0,t.refreshWidgets(r,!0,!0),t.restoreHeaders(r),u(r),t.bindEvents(r,s.$headers,!0),v(r),p(r,a,n)})).bind("update"+s.namespace+" updateRows"+s.namespace,(function(e,t,s){e.stopPropagation(),r.isUpdating=!0,f(r),p(r,t,s)})).bind("updateCell"+s.namespace,(function(t,a,i,d){t.stopPropagation(),r.isUpdating=!0,o.find(s.selectorRemove).remove();var c,l,u,p,f=o.find("tbody"),g=e(a),h=f.index(e.fn.closest?g.closest("tbody"):g.parents("tbody").filter(":first")),m=e.fn.closest?g.closest("tr"):g.parents("tr").filter(":first");a=g[0],f.length&&h>=0&&(u=f.eq(h).find("tr").index(m),p=g.index(),s.cache[h].normalized[u][s.columns].$row=m,l=void 0===s.extractors[p].id?n(r,a,p):s.extractors[p].format(n(r,a,p),r,a,p),c="no-parser"===s.parsers[p].id?"":s.parsers[p].format(l,r,a,p),s.cache[h].normalized[u][p]=s.ignoreCase&&"string"==typeof c?c.toLowerCase():c,"numeric"===(s.parsers[p].type||"").toLowerCase()&&(s.cache[h].colMax[p]=Math.max(Math.abs(c)||0,s.cache[h].colMax[p]||0)),w(o,i,d))})).bind("addRows"+s.namespace,(function(t,d,c,l){if(t.stopPropagation(),r.isUpdating=!0,a(s.cache))f(r),p(r,c,l);else{var u,g,h,m,b,y,v,x=(d=e(d).attr("role","row")).filter("tr").length,C=o.find("tbody").index(d.parents("tbody").filter(":first"));for(s.parsers&&s.parsers.length||i(r),u=0;u<x;u++){for(h=d[u].cells.length,v=[],y={child:[],$row:d.eq(u),order:s.cache[C].normalized.length},g=0;g<h;g++)m=void 0===s.extractors[g].id?n(r,d[u].cells[g],g):s.extractors[g].format(n(r,d[u].cells[g],g),r,d[u].cells[g],g),b="no-parser"===s.parsers[g].id?"":s.parsers[g].format(m,r,d[u].cells[g],g),v[g]=s.ignoreCase&&"string"==typeof b?b.toLowerCase():b,"numeric"===(s.parsers[g].type||"").toLowerCase()&&(s.cache[C].colMax[g]=Math.max(Math.abs(v[g])||0,s.cache[C].colMax[g]||0));v.push(y),s.cache[C].normalized.push(v)}w(o,c,l)}})).bind("updateComplete"+s.namespace,(function(){r.isUpdating=!1})).bind("sorton"+s.namespace,(function(s,n,i,l){var u=r.config;s.stopPropagation(),o.trigger("sortStart",this),function(t,r){var s,a,n,o,i,d=t.config,c=r||d.sortList;d.sortList=[],e.each(c,(function(t,r){if(o=parseInt(r[0],10),n=d.$headers.filter('[data-column="'+o+'"]:last')[0]){switch(a=(a=(""+r[1]).match(/^(1|d|s|o|n)/))?a[0]:""){case"1":case"d":a=1;break;case"s":a=i||0;break;case"o":s=n.order[(i||0)%(d.sortReset?3:2)],a=0===s?1:1===s?0:2;break;case"n":n.count=n.count+1,a=n.order[n.count%(d.sortReset?3:2)];break;default:a=0}i=0===t?a:i,s=[o,parseInt(a,10)||0],d.sortList.push(s),a=e.inArray(s[1],n.order),n.count=a>=0?a:s[1]%(d.sortReset?3:2)}}))}(r,n),g(r),u.delayInit&&a(u.cache)&&d(r),o.trigger("sortBegin",this),b(r),c(r,l),o.trigger("sortEnd",this),t.applyWidget(r),e.isFunction(i)&&i(r)})).bind("appendCache"+s.namespace,(function(t,s,a){t.stopPropagation(),c(r,a),e.isFunction(s)&&s(r)})).bind("updateCache"+s.namespace,(function(t,a){s.parsers&&s.parsers.length||i(r),d(r),e.isFunction(a)&&a(r)})).bind("applyWidgetId"+s.namespace,(function(e,a){e.stopPropagation(),t.getWidgetById(a).format(r,s,s.widgetOptions)})).bind("applyWidgets"+s.namespace,(function(e,s){e.stopPropagation(),t.applyWidget(r,s)})).bind("refreshWidgets"+s.namespace,(function(e,s,a){e.stopPropagation(),t.refreshWidgets(r,s,a)})).bind("destroy"+s.namespace,(function(e,s,a){e.stopPropagation(),t.destroy(r,s,a)})).bind("resetToLoadState"+s.namespace,(function(){t.refreshWidgets(r,!0,!0),s=e.extend(!0,t.defaults,s.originalSettings),r.hasInitialized=!1,t.setup(r,s)}))}t.version="2.17.7",t.parsers=[],t.widgets=[],t.defaults={theme:"default",widthFixed:!1,showProcessing:!1,headerTemplate:"{content}",onRenderTemplate:null,onRenderHeader:null,cancelSelection:!0,tabIndex:!0,dateFormat:"mmddyyyy",sortMultiSortKey:"shiftKey",sortResetKey:"ctrlKey",usNumberFormat:!0,delayInit:!1,serverSideSorting:!1,headers:{},ignoreCase:!0,sortForce:null,sortList:[],sortAppend:null,sortStable:!1,sortInitialOrder:"asc",sortLocaleCompare:!1,sortReset:!1,sortRestart:!1,emptyTo:"bottom",stringTo:"max",textExtraction:"basic",textAttribute:"data-text",textSorter:null,numberSorter:null,widgets:[],widgetOptions:{zebra:["even","odd"]},initWidgets:!0,initialized:null,tableClass:"",cssAsc:"",cssDesc:"",cssNone:"",cssHeader:"",cssHeaderRow:"",cssProcessing:"",cssChildRow:"tablesorter-childRow",cssIcon:"tablesorter-icon",cssInfoBlock:"tablesorter-infoOnly",selectorHeaders:"> thead th, > thead td",selectorSort:"th, td",selectorRemove:".remove-me",debug:!1,headerList:[],empties:{},strings:{},parsers:[]},t.css={table:"tablesorter",cssHasChild:"tablesorter-hasChildRow",childRow:"tablesorter-childRow",header:"tablesorter-header",headerRow:"tablesorter-headerRow",headerIn:"tablesorter-header-inner",icon:"tablesorter-icon",info:"tablesorter-infoOnly",processing:"tablesorter-processing",sortAsc:"tablesorter-headerAsc",sortDesc:"tablesorter-headerDesc",sortNone:"tablesorter-headerUnSorted"},t.language={sortAsc:"Ascending sort applied, ",sortDesc:"Descending sort applied, ",sortNone:"No sort applied, ",nextAsc:"activate to apply an ascending sort",nextDesc:"activate to apply a descending sort",nextNone:"activate to remove the sort"},t.log=r,t.benchmark=s,t.construct=function(r){return this.each((function(){var s=this,a=e.extend(!0,{},t.defaults,r);a.originalSettings=r,!s.hasInitialized&&t.buildTable&&"TABLE"!==this.tagName?t.buildTable(s,a):t.setup(s,a)}))},t.setup=function(s,a){if(!s||!s.tHead||0===s.tBodies.length||!0===s.hasInitialized)return a.debug?r("ERROR: stopping initialization! No table, thead, tbody or tablesorter has already been initialized"):"";var n,o="",c=e(s),l=e.metadata;s.hasInitialized=!1,s.isProcessing=!0,s.config=a,e.data(s,"tablesorter",a),a.debug&&e.data(s,"startoveralltimer",new Date),a.supportsDataObject=((n=e.fn.jquery.split("."))[0]=parseInt(n[0],10),n[0]>1||1===n[0]&&parseInt(n[1],10)>=4),a.string={max:1,min:-1,emptyMin:1,emptyMax:-1,zero:0,none:0,null:0,top:!0,bottom:!1},/tablesorter\-/.test(c.attr("class"))||(o=""!==a.theme?" tablesorter-"+a.theme:""),a.table=s,a.$table=c.addClass(t.css.table+" "+a.tableClass+o).attr("role","grid"),a.$headers=c.find(a.selectorHeaders),a.namespace?a.namespace="."+a.namespace.replace(/\W/g,""):a.namespace=".tablesorter"+Math.random().toString(16).slice(2),a.$table.children().children("tr").attr("role","row"),a.$tbodies=c.children("tbody:not(."+a.cssInfoBlock+")").attr({"aria-live":"polite","aria-relevant":"all"}),a.$table.find("caption").length&&a.$table.attr("aria-labelledby","theCaption"),a.widgetInit={},a.textExtraction=a.$table.attr("data-text-extraction")||a.textExtraction||"basic",u(s),function(t){if(t.config.widthFixed&&0===e(t).find("colgroup").length){var r=e("<colgroup>"),s=e(t).width();e(t.tBodies[0]).find("tr:first").children(":visible").each((function(){r.append(e("<col>").css("width",parseInt(e(this).width()/s*1e3,10)/10+"%"))})),e(t).prepend(r)}}(s),i(s),a.totalRows=0,a.delayInit||d(s),t.bindEvents(s,a.$headers,!0),v(s),a.supportsDataObject&&void 0!==c.data().sortlist?a.sortList=c.data().sortlist:l&&c.metadata()&&c.metadata().sortlist&&(a.sortList=c.metadata().sortlist),t.applyWidget(s,!0),a.sortList.length>0?c.trigger("sorton",[a.sortList,{},!a.initWidgets,!0]):(g(s),a.initWidgets&&t.applyWidget(s,!1)),a.showProcessing&&c.unbind("sortBegin"+a.namespace+" sortEnd"+a.namespace).bind("sortBegin"+a.namespace+" sortEnd"+a.namespace,(function(e){clearTimeout(a.processTimer),t.isProcessing(s),"sortBegin"===e.type&&(a.processTimer=setTimeout((function(){t.isProcessing(s,!0)}),500))})),s.hasInitialized=!0,s.isProcessing=!1,a.debug&&t.benchmark("Overall initialization time",e.data(s,"startoveralltimer")),c.trigger("tablesorter-initialized",s),"function"==typeof a.initialized&&a.initialized(s)},t.getColumnData=function(t,r,s,a){if(null!=r){var n,o=(t=e(t)[0]).config;if(r[s])return a?r[s]:r[o.$headers.index(o.$headers.filter('[data-column="'+s+'"]:last'))];for(n in r)if("string"==typeof n&&(a?o.$headers.eq(s).filter(n):o.$headers.filter('[data-column="'+s+'"]:last').filter(n)).length)return r[n]}},t.computeColumnIndex=function(t){var r,s,a,n,o,i,d,c,l,u,p,f,g,h=[],m={},b=0;for(r=0;r<t.length;r++)for(d=t[r].cells,s=0;s<d.length;s++){for(i=d[s],o=e(i),l=(c=i.parentNode.rowIndex)+"-"+o.index(),u=i.rowSpan||1,p=i.colSpan||1,void 0===h[c]&&(h[c]=[]),a=0;a<h[c].length+1;a++)if(void 0===h[c][a]){f=a;break}for(m[l]=f,b=Math.max(f,b),o.attr({"data-column":f}),a=c;a<c+u;a++)for(void 0===h[a]&&(h[a]=[]),g=h[a],n=f;n<f+p;n++)g[n]="x"}return b+1},t.isProcessing=function(r,s,a){var n=(r=e(r))[0].config,o=a||r.find("."+t.css.header);s?(void 0!==a&&n.sortList.length>0&&(o=o.filter((function(){return!this.sortDisabled&&t.isValueInArray(parseFloat(e(this).attr("data-column")),n.sortList)>=0}))),r.add(o).addClass(t.css.processing+" "+n.cssProcessing)):r.add(o).removeClass(t.css.processing+" "+n.cssProcessing)},t.processTbody=function(t,r,s){var a;if(t=e(t)[0],s)return t.isProcessing=!0,r.before('<span class="tablesorter-savemyplace"/>'),a=e.fn.detach?r.detach():r.remove();a=e(t).find("span.tablesorter-savemyplace"),r.insertAfter(a),a.remove(),t.isProcessing=!1},t.clearTableBody=function(t){e(t)[0].config.$tbodies.children().detach()},t.bindEvents=function(t,r,s){var n,o=(t=e(t)[0]).config;!0!==s&&(o.$extraHeaders=o.$extraHeaders?o.$extraHeaders.add(r):r),r.find(o.selectorSort).add(r.filter(o.selectorSort)).unbind("mousedown mouseup sort keyup ".split(" ").join(o.namespace+" ")).bind("mousedown mouseup sort keyup ".split(" ").join(o.namespace+" "),(function(s,i){var c,l=s.type;if(!(1!==(s.which||s.button)&&!/sort|keyup/.test(l)||"keyup"===l&&13!==s.which||"mouseup"===l&&!0!==i&&(new Date).getTime()-n>250)){if("mousedown"===l)return n=(new Date).getTime(),/(input|select|button|textarea)/i.test(s.target.tagName)?"":!o.cancelSelection;o.delayInit&&a(o.cache)&&d(t),c=e.fn.closest?e(this).closest("th, td")[0]:/TH|TD/.test(this.tagName)?this:e(this).parents("th, td")[0],(c=o.$headers[r.index(c)]).sortDisabled||m(t,c,s)}})),o.cancelSelection&&r.attr("unselectable","on").bind("selectstart",!1).css({"user-select":"none",MozUserSelect:"none"})},t.restoreHeaders=function(r){var s=e(r)[0].config;s.$table.find(s.selectorHeaders).each((function(r){e(this).find("."+t.css.headerIn).length&&e(this).html(s.headerContent[r])}))},t.destroy=function(r,s,a){if((r=e(r)[0]).hasInitialized){t.refreshWidgets(r,!0,!0);var n=e(r),o=r.config,i=n.find("thead:first"),d=i.find("tr."+t.css.headerRow).removeClass(t.css.headerRow+" "+o.cssHeaderRow),c=n.find("tfoot:first > tr").children("th, td");!1===s&&e.inArray("uitheme",o.widgets)>=0&&(n.trigger("applyWidgetId",["uitheme"]),n.trigger("applyWidgetId",["zebra"])),i.find("tr").not(d).remove(),n.removeData("tablesorter").unbind("sortReset update updateAll updateRows updateCell addRows updateComplete sorton appendCache updateCache applyWidgetId applyWidgets refreshWidgets destroy mouseup mouseleave keypress sortBegin sortEnd resetToLoadState ".split(" ").join(o.namespace+" ")),o.$headers.add(c).removeClass([t.css.header,o.cssHeader,o.cssAsc,o.cssDesc,t.css.sortAsc,t.css.sortDesc,t.css.sortNone].join(" ")).removeAttr("data-column").removeAttr("aria-label").attr("aria-disabled","true"),d.find(o.selectorSort).unbind("mousedown mouseup keypress ".split(" ").join(o.namespace+" ")),t.restoreHeaders(r),n.toggleClass(t.css.table+" "+o.tableClass+" tablesorter-"+o.theme,!1===s),r.hasInitialized=!1,delete r.config.cache,"function"==typeof a&&a(r)}},t.regex={chunk:/(^([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)?$|^0x[0-9a-f]+$|\d+)/gi,chunks:/(^\\0|\\0$)/,hex:/^0x[0-9a-f]+$/i},t.sortNatural=function(e,r){if(e===r)return 0;var s,a,n,o,i,d,c,l,u=t.regex;if(u.hex.test(r)){if((a=parseInt(e.match(u.hex),16))<(o=parseInt(r.match(u.hex),16)))return-1;if(a>o)return 1}for(s=e.replace(u.chunk,"\\0$1\\0").replace(u.chunks,"").split("\\0"),n=r.replace(u.chunk,"\\0$1\\0").replace(u.chunks,"").split("\\0"),l=Math.max(s.length,n.length),c=0;c<l;c++){if(i=isNaN(s[c])?s[c]||0:parseFloat(s[c])||0,d=isNaN(n[c])?n[c]||0:parseFloat(n[c])||0,isNaN(i)!==isNaN(d))return isNaN(i)?1:-1;if(typeof i!=typeof d&&(i+="",d+=""),i<d)return-1;if(i>d)return 1}return 0},t.sortNaturalAsc=function(e,r,s,a,n){if(e===r)return 0;var o=n.string[n.empties[s]||n.emptyTo];return""===e&&0!==o?"boolean"==typeof o?o?-1:1:-o||-1:""===r&&0!==o?"boolean"==typeof o?o?1:-1:o||1:t.sortNatural(e,r)},t.sortNaturalDesc=function(e,r,s,a,n){if(e===r)return 0;var o=n.string[n.empties[s]||n.emptyTo];return""===e&&0!==o?"boolean"==typeof o?o?-1:1:o||1:""===r&&0!==o?"boolean"==typeof o?o?1:-1:-o||-1:t.sortNatural(r,e)},t.sortText=function(e,t){return e>t?1:e<t?-1:0},t.getTextValue=function(e,t,r){if(r){var s,a=e?e.length:0,n=r+t;for(s=0;s<a;s++)n+=e.charCodeAt(s);return t*n}return 0},t.sortNumericAsc=function(e,r,s,a,n,o){if(e===r)return 0;var i=o.config,d=i.string[i.empties[n]||i.emptyTo];return""===e&&0!==d?"boolean"==typeof d?d?-1:1:-d||-1:""===r&&0!==d?"boolean"==typeof d?d?1:-1:d||1:(isNaN(e)&&(e=t.getTextValue(e,s,a)),isNaN(r)&&(r=t.getTextValue(r,s,a)),e-r)},t.sortNumericDesc=function(e,r,s,a,n,o){if(e===r)return 0;var i=o.config,d=i.string[i.empties[n]||i.emptyTo];return""===e&&0!==d?"boolean"==typeof d?d?-1:1:d||1:""===r&&0!==d?"boolean"==typeof d?d?1:-1:-d||-1:(isNaN(e)&&(e=t.getTextValue(e,s,a)),isNaN(r)&&(r=t.getTextValue(r,s,a)),r-e)},t.sortNumeric=function(e,t){return e-t},t.characterEquivalents={a:"áàâãäąå",A:"ÁÀÂÃÄĄÅ",c:"çćč",C:"ÇĆČ",e:"éèêëěę",E:"ÉÈÊËĚĘ",i:"íìİîïı",I:"ÍÌİÎÏ",o:"óòôõö",O:"ÓÒÔÕÖ",ss:"ß",SS:"ẞ",u:"úùûüů",U:"ÚÙÛÜŮ"},t.replaceAccents=function(e){var r,s="[",a=t.characterEquivalents;if(!t.characterRegex){for(r in t.characterRegexArray={},a)"string"==typeof r&&(s+=a[r],t.characterRegexArray[r]=new RegExp("["+a[r]+"]","g"));t.characterRegex=new RegExp(s+"]")}if(t.characterRegex.test(e))for(r in a)"string"==typeof r&&(e=e.replace(t.characterRegexArray[r],r));return e},t.isValueInArray=function(e,t){var r,s=t.length;for(r=0;r<s;r++)if(t[r][0]===e)return r;return-1},t.addParser=function(e){var r,s=t.parsers.length,a=!0;for(r=0;r<s;r++)t.parsers[r].id.toLowerCase()===e.id.toLowerCase()&&(a=!1);a&&t.parsers.push(e)},t.getParserById=function(e){if("false"==e)return!1;var r,s=t.parsers.length;for(r=0;r<s;r++)if(t.parsers[r].id.toLowerCase()===e.toString().toLowerCase())return t.parsers[r];return!1},t.addWidget=function(e){t.widgets.push(e)},t.hasWidget=function(t,r){return(t=e(t)).length&&t[0].config&&t[0].config.widgetInit[r]||!1},t.getWidgetById=function(e){var r,s,a=t.widgets.length;for(r=0;r<a;r++)if((s=t.widgets[r])&&s.hasOwnProperty("id")&&s.id.toLowerCase()===e.toLowerCase())return s},t.applyWidget=function(r,a){var n,o,i,d=(r=e(r)[0]).config,c=d.widgetOptions,l=[];!1!==a&&r.hasInitialized&&(r.isApplyingWidgets||r.isUpdating)||(d.debug&&(n=new Date),d.widgets.length&&(r.isApplyingWidgets=!0,d.widgets=e.grep(d.widgets,(function(t,r){return e.inArray(t,d.widgets)===r})),e.each(d.widgets||[],(function(e,r){(i=t.getWidgetById(r))&&i.id&&(i.priority||(i.priority=10),l[e]=i)})),l.sort((function(e,t){return e.priority<t.priority?-1:e.priority===t.priority?0:1})),e.each(l,(function(t,s){s&&(!a&&d.widgetInit[s.id]||(d.widgetInit[s.id]=!0,s.hasOwnProperty("options")&&(c=r.config.widgetOptions=e.extend(!0,{},s.options,c)),s.hasOwnProperty("init")&&s.init(r,s,d,c)),!a&&s.hasOwnProperty("format")&&s.format(r,d,c,!1))}))),setTimeout((function(){r.isApplyingWidgets=!1}),0),d.debug&&(o=d.widgets.length,s("Completed "+(!0===a?"initializing ":"applying ")+o+" widget"+(1!==o?"s":""),n)))},t.refreshWidgets=function(s,a,n){var o,i=(s=e(s)[0]).config,d=i.widgets,c=t.widgets,l=c.length;for(o=0;o<l;o++)c[o]&&c[o].id&&(a||e.inArray(c[o].id,d)<0)&&(i.debug&&r('Refeshing widgets: Removing "'+c[o].id+'"'),c[o].hasOwnProperty("remove")&&i.widgetInit[c[o].id]&&(c[o].remove(s,i,i.widgetOptions),i.widgetInit[c[o].id]=!1));!0!==n&&t.applyWidget(s,a)},t.getData=function(t,r,s){var a,n,o="",i=e(t);return i.length?(a=!!e.metadata&&i.metadata(),n=" "+(i.attr("class")||""),void 0!==i.data(s)||void 0!==i.data(s.toLowerCase())?o+=i.data(s)||i.data(s.toLowerCase()):a&&void 0!==a[s]?o+=a[s]:r&&void 0!==r[s]?o+=r[s]:" "!==n&&n.match(" "+s+"-")&&(o=n.match(new RegExp("\\s"+s+"-([\\w-]+)"))[1]||""),e.trim(o)):""},t.formatFloat=function(t,r){return"string"!=typeof t||""===t?t:(t=(r&&r.config?!1!==r.config.usNumberFormat:void 0===r||r)?t.replace(/,/g,""):t.replace(/[\s|\.]/g,"").replace(/,/g,"."),/^\s*\([.\d]+\)/.test(t)&&(t=t.replace(/^\s*\(([.\d]+)\)/,"-$1")),s=parseFloat(t),isNaN(s)?e.trim(t):s);var s},t.isDigit=function(e){return!isNaN(e)||/^[\-+(]?\d+[)]?$/.test(e.toString().replace(/[,.'"\s]/g,""))}}});var t=e.tablesorter;e.fn.extend({tablesorter:t.construct}),t.addParser({id:"no-parser",is:function(){return!1},format:function(){return""},type:"text"}),t.addParser({id:"text",is:function(){return!0},format:function(r,s){var a=s.config;return r&&(r=e.trim(a.ignoreCase?r.toLocaleLowerCase():r),r=a.sortLocaleCompare?t.replaceAccents(r):r),r},type:"text"}),t.addParser({id:"digit",is:function(e){return t.isDigit(e)},format:function(r,s){var a=t.formatFloat((r||"").replace(/[^\w,. \-()]/g,""),s);return r&&"number"==typeof a?a:r?e.trim(r&&s.config.ignoreCase?r.toLocaleLowerCase():r):r},type:"numeric"}),t.addParser({id:"currency",is:function(e){return/^\(?\d+[\u00a3$\u20ac\u00a4\u00a5\u00a2?.]|[\u00a3$\u20ac\u00a4\u00a5\u00a2?.]\d+\)?$/.test((e||"").replace(/[+\-,. ]/g,""))},format:function(r,s){var a=t.formatFloat((r||"").replace(/[^\w,. \-()]/g,""),s);return r&&"number"==typeof a?a:r?e.trim(r&&s.config.ignoreCase?r.toLocaleLowerCase():r):r},type:"numeric"}),t.addParser({id:"ipAddress",is:function(e){return/^\d{1,3}[\.]\d{1,3}[\.]\d{1,3}[\.]\d{1,3}$/.test(e)},format:function(e,r){var s,a=e?e.split("."):"",n="",o=a.length;for(s=0;s<o;s++)n+=("00"+a[s]).slice(-3);return e?t.formatFloat(n,r):e},type:"numeric"}),t.addParser({id:"url",is:function(e){return/^(https?|ftp|file):\/\//.test(e)},format:function(t){return t?e.trim(t.replace(/(https?|ftp|file):\/\//,"")):t},type:"text"}),t.addParser({id:"isoDate",is:function(e){return/^\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}/.test(e)},format:function(e,r){return e?t.formatFloat(""!==e?new Date(e.replace(/-/g,"/")).getTime()||e:"",r):e},type:"numeric"}),t.addParser({id:"percent",is:function(e){return/(\d\s*?%|%\s*?\d)/.test(e)&&e.length<15},format:function(e,r){return e?t.formatFloat(e.replace(/%/g,""),r):e},type:"numeric"}),t.addParser({id:"usLongDate",is:function(e){return/^[A-Z]{3,10}\.?\s+\d{1,2},?\s+(\d{4})(\s+\d{1,2}:\d{2}(:\d{2})?(\s+[AP]M)?)?$/i.test(e)||/^\d{1,2}\s+[A-Z]{3,10}\s+\d{4}/i.test(e)},format:function(e,r){return e?t.formatFloat(new Date(e.replace(/(\S)([AP]M)$/i,"$1 $2")).getTime()||e,r):e},type:"numeric"}),t.addParser({id:"shortDate",is:function(e){return/(^\d{1,2}[\/\s]\d{1,2}[\/\s]\d{4})|(^\d{4}[\/\s]\d{1,2}[\/\s]\d{1,2})/.test((e||"").replace(/\s+/g," ").replace(/[\-.,]/g,"/"))},format:function(e,r,s,a){if(e){var n=r.config,o=n.$headers.filter("[data-column="+a+"]:last"),i=o.length&&o[0].dateFormat||t.getData(o,t.getColumnData(r,n.headers,a),"dateFormat")||n.dateFormat;e=e.replace(/\s+/g," ").replace(/[\-.,]/g,"/"),"mmddyyyy"===i?e=e.replace(/(\d{1,2})[\/\s](\d{1,2})[\/\s](\d{4})/,"$3/$1/$2"):"ddmmyyyy"===i?e=e.replace(/(\d{1,2})[\/\s](\d{1,2})[\/\s](\d{4})/,"$3/$2/$1"):"yyyymmdd"===i&&(e=e.replace(/(\d{4})[\/\s](\d{1,2})[\/\s](\d{1,2})/,"$1/$2/$3"))}return e?t.formatFloat(new Date(e).getTime()||e,r):e},type:"numeric"}),t.addParser({id:"time",is:function(e){return/^(([0-2]?\d:[0-5]\d)|([0-1]?\d:[0-5]\d\s?([AP]M)))$/i.test(e)},format:function(e,r){return e?t.formatFloat(new Date("2000/01/01 "+e.replace(/(\S)([AP]M)$/i,"$1 $2")).getTime()||e,r):e},type:"numeric"}),t.addParser({id:"metadata",is:function(){return!1},format:function(t,r,s){var a=r.config,n=a.parserMetadataName?a.parserMetadataName:"sortValue";return e(s).metadata()[n]},type:"numeric"}),t.addWidget({id:"zebra",priority:90,format:function(r,s,a){var n,o,i,d,c,l,u=new RegExp(s.cssChildRow,"i"),p=s.$tbodies;for(s.debug&&(c=new Date),l=0;l<p.length;l++)(n=p.eq(l)).children("tr").length>1&&(i=0,n.children("tr:visible").not(s.selectorRemove).each((function(){o=e(this),u.test(this.className)||i++,d=i%2==0,o.removeClass(a.zebra[d?1:0]).addClass(a.zebra[d?0:1])})));s.debug&&t.benchmark("Applying Zebra widget",c)},remove:function(t,r,s){var a,n,o=r.$tbodies,i=(s.zebra||["even","odd"]).join(" ");for(a=0;a<o.length;a++)(n=e.tablesorter.processTbody(t,o.eq(a),!0)).children().removeClass(i),e.tablesorter.processTbody(t,n,!1)}})}(jQuery)},M8Tw:function(e,t,r){"use strict";r.r(t);r("jJ4L");var s=r("+x/D"),a=(r("AqzS"),r("KloK"));const n={sortMultiSortKey:"",headers:{},debug:!1,tabIndex:!1};function o(e,t){let r=Object.assign({},n,t);return e.find("th").each((function(e,t){const a=Object(s.default)(t);r.headers[e]={},a.hasClass("aui-table-column-unsortable")?r.headers[e].sorter=!1:(a.attr("tabindex","0"),a.wrapInner("<span class='aui-table-header-content'/>"),a.hasClass("aui-table-column-issue-key")&&(r.headers[e].sorter="issue-key"))})),e.tablesorter(r)}let i={setup:function(){s.default.tablesorter.addParser({id:"issue-key",is:function(){return!1},format:function(e){const t=e.split("-"),r=t[0],s=t[1],a="..........",n="000000";let o=(r+a).slice(0,a.length);return o+=(n+s).slice(-n.length),o},type:"text"}),s.default.tablesorter.addParser({id:"textSortAttributeParser",is:function(e,t,r){return r.hasAttribute("data-sort-value")&&(!r.hasAttribute("data-sort-type")||"text"===r.getAttribute("data-sort-type"))},format:function(e,t,r){return r.getAttribute("data-sort-value")},type:"text"}),s.default.tablesorter.addParser({id:"numericSortAttributeParser",is:function(e,t,r){return"numeric"===r.getAttribute("data-sort-type")&&r.hasAttribute("data-sort-value")},format:function(e,t,r){return r.getAttribute("data-sort-value")},type:"numeric"}),Object(s.default)(".aui-table-sortable").each((function(){o(Object(s.default)(this))}))},setTableSortable:function(e,t){return o(e,t)}};Object(s.default)(i.setup),Object(a.default)("tablessortable",i)}},[["M8Tw","runtime","aui.splitchunk.0d131bcbf1","aui.splitchunk.fbbef27525","aui.splitchunk.444efc83be","aui.splitchunk.739b9ec8cc","aui.splitchunk.056561461c","aui.splitchunk.949297951c","aui.splitchunk.d7c46c2734","aui.splitchunk.fb15cffa72","aui.splitchunk.862f26d10a","aui.splitchunk.b2ecdd4179","aui.splitchunk.16f099a0da"]]]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:tasks-report-resources', location = 'templates/tasks-report.soy' */
// This file was automatically generated from tasks-report.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.InlineTasks.Report.Templates.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.InlineTasks == 'undefined') { Confluence.InlineTasks = {}; }
if (typeof Confluence.InlineTasks.Report == 'undefined') { Confluence.InlineTasks.Report = {}; }
if (typeof Confluence.InlineTasks.Report.Templates == 'undefined') { Confluence.InlineTasks.Report.Templates = {}; }


Confluence.InlineTasks.Report.Templates.tasksReport = function(opt_data, opt_ignored) {
  var output = '<div class="table-wrapper" data-loading="false"><div class="task-blanket"></div><input type="hidden" name="reportParameters" value="' + soy.$$escapeHtml(opt_data.reportParameters) + '" /><table class="aui aui-table-interactive tasks-report" data-sortable="false" data-total-pages="' + soy.$$escapeHtml(opt_data.totalPages) + '" data-page-size="' + soy.$$escapeHtml(opt_data.pageSize) + '" data-adaptive="' + soy.$$escapeHtml(opt_data.adaptive) + '" data-page-limit="' + soy.$$escapeHtml(opt_data.pageLimit) + '"><thead><tr class="tablesorter-headerRow">';
  var headingList14 = opt_data.headings;
  var headingListLen14 = headingList14.length;
  for (var headingIndex14 = 0; headingIndex14 < headingListLen14; headingIndex14++) {
    var headingData14 = headingList14[headingIndex14];
    output += '<th class="header-' + soy.$$escapeHtml(headingData14) + ((headingData14 == 'description') ? ' aui-table-column-unsortable' : '') + '" data-column-name="' + soy.$$escapeHtml(headingData14) + '">' + soy.$$escapeHtml(opt_data.headingTexts[headingData14]) + '</th>';
  }
  output += '</tr></thead><tbody>';
  if (opt_data.tasks.length) {
    var taskList29 = opt_data.tasks;
    var taskListLen29 = taskList29.length;
    for (var taskIndex29 = 0; taskIndex29 < taskListLen29; taskIndex29++) {
      var taskData29 = taskList29[taskIndex29];
      output += Confluence.InlineTasks.Report.Templates.tasksReportLine({task: taskData29, columns: opt_data.headings});
    }
  } else {
    output += '<tr><td colspan="' + soy.$$escapeHtml(opt_data.headings.length) + '">' + soy.$$escapeHtml('Create a task list in a Confluence page to keep track of things you need to do.') + '</td></tr>';
  }
  output += '</tbody></table></div>';
  return output;
};
if (goog.DEBUG) {
  Confluence.InlineTasks.Report.Templates.tasksReport.soyTemplateName = 'Confluence.InlineTasks.Report.Templates.tasksReport';
}


Confluence.InlineTasks.Report.Templates.tasksFeatureDiscoveryNotification = function(opt_data, opt_ignored) {
  return '<h2>' + soy.$$escapeHtml('Hey! Did you know...') + '</h2><p>' + soy.$$escapeHtml('You can view the tasks you created, or assigned to you, in the tasks tabs of your profile.') + '</p>';
};
if (goog.DEBUG) {
  Confluence.InlineTasks.Report.Templates.tasksFeatureDiscoveryNotification.soyTemplateName = 'Confluence.InlineTasks.Report.Templates.tasksFeatureDiscoveryNotification';
}


Confluence.InlineTasks.Report.Templates.tasksReportLine = function(opt_data, opt_ignored) {
  var output = '<tr data-task-id="' + soy.$$escapeHtml(opt_data.task.taskId) + '" aria-disabled="false">';
  var columnList51 = opt_data.columns;
  var columnListLen51 = columnList51.length;
  for (var columnIndex51 = 0; columnIndex51 < columnListLen51; columnIndex51++) {
    var columnData51 = columnList51[columnIndex51];
    if (columnData51 == 'duedate') {
      output += '<td class=\'tasks-report-date\'>' + ((opt_data.task.dueDate) ? soy.$$escapeHtml(opt_data.task.dueDate) : '') + '</td>';
    } else if (columnData51 == 'description') {
      output += '<td>' + soy.$$filterNoAutoescape(opt_data.task.taskHtml) + '</td>';
    } else if (columnData51 == 'assignee') {
      output += '<td class=\'tasks-report-assignee\'>' + ((opt_data.task.assigneeUserName) ? Confluence.Templates.User.usernameLink({username: opt_data.task.assigneeUserName, fullName: opt_data.task.assigneeFullName, canView: false}) : '') + '</td>';
    } else if (columnData51 == 'location') {
      output += '<td><a class=\'task-location\' href="' + soy.$$escapeHtml("/wiki") + soy.$$escapeHtml(opt_data.task.pageUrl) + '">' + soy.$$escapeHtml(opt_data.task.pageTitle) + '</a></td>';
    } else if (columnData51 == 'completedate') {
      output += '<td class=\'tasks-report-date\'>' + ((opt_data.task.completeDate) ? soy.$$escapeHtml(opt_data.task.completeDate) : (opt_data.task.taskCompleted) ? '<span class="emptycompletedate">--</span>' : '') + '</td>';
    } else if (columnData51 == 'labels') {
      output += '<td>';
      var labelList90 = opt_data.task.labels;
      var labelListLen90 = labelList90.length;
      for (var labelIndex90 = 0; labelIndex90 < labelListLen90; labelIndex90++) {
        var labelData90 = labelList90[labelIndex90];
        output += aui.labels.label({text: labelData90});
      }
      output += '</td>';
    }
  }
  output += '</tr>';
  return output;
};
if (goog.DEBUG) {
  Confluence.InlineTasks.Report.Templates.tasksReportLine.soyTemplateName = 'Confluence.InlineTasks.Report.Templates.tasksReportLine';
}


Confluence.InlineTasks.Report.Templates.taskReportBrowserWarning = function(opt_data, opt_ignored) {
  var param98 = '<p>' + soy.$$escapeHtml('Unable to show this task report.') + '</p>';
  var messageList102 = opt_data.messages;
  var messageListLen102 = messageList102.length;
  for (var messageIndex102 = 0; messageIndex102 < messageListLen102; messageIndex102++) {
    var messageData102 = messageList102[messageIndex102];
    param98 += '<p>' + soy.$$escapeHtml(messageData102) + '</p>';
  }
  var output = '' + aui.message.warning({content: param98});
  return output;
};
if (goog.DEBUG) {
  Confluence.InlineTasks.Report.Templates.taskReportBrowserWarning.soyTemplateName = 'Confluence.InlineTasks.Report.Templates.taskReportBrowserWarning';
}


Confluence.InlineTasks.Report.Templates.taskReportWarning = function(opt_data, opt_ignored) {
  return '' + aui.message.warning({content: '<p>' + soy.$$escapeHtml('Unable to show the task report. Edit this page to resolve the issues.') + '</p>'});
};
if (goog.DEBUG) {
  Confluence.InlineTasks.Report.Templates.taskReportWarning.soyTemplateName = 'Confluence.InlineTasks.Report.Templates.taskReportWarning';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:tasks-report-resources', location = 'js/tasks-report-blank-exp.js' */
(function(d){Confluence.InlineTasks=Confluence.InlineTasks||{};Confluence.InlineTasks.TasksReport=Confluence.InlineTasks.TasksReport||{};var b={"blank.complete.title":"Task report","blank.complete.desc":"Get going, no tasks completed yet.","blank.incomplete.title":"Task report","blank.incomplete.desc":"Looking good, no incomplete tasks."};Confluence.InlineTasks.TasksReport.renderBlankExperiences=function(c,
a){a||(a="incomplete");a=Confluence.UI.Components.BlankPlaceholderBox.Templates.blankBox({blankTitle:b["blank."+a+".title"],blankDescription:b["blank."+a+".desc"],customClass:a+" tasks-report-blank"});c.html(a)}})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:tasks-report-resources', location = 'js/tasks-report.js' */
AJS.$(function(c){var l=Confluence.InlineTasks.Report.Templates,g=Confluence.InlineTasks.TasksTableSortable,f=c(".tasks-report").parent();_.each(f,function(d){var e=c(d),a=e.find("table.tasks-report"),f=a.data("page-size");d=a.data("total-pages");var m=a.data("page-limit"),h=a.data("adaptive");e=e.find("input[name\x3dreportParameters]").val();var b=JSON.parse(e);AJS.bind("inline-tasks.status-update.start",function(a,b){0<b.taskListQueue.length&&(a=c("li[data-inline-task-id\x3d"+b.taskId+"]").closest(".tasks-report").siblings(".macro-auto-pagination"),
c(a).attr("aria-disabled",!0),a=a.find("li a"),a.on("click.taskreport.pagination",function(a){a.preventDefault();a.stopPropagation()}))});AJS.bind("inline-tasks.status-update.complete",function(a,b){0===b.taskListQueue.length&&(c(".macro-auto-pagination").attr("aria-disabled",!1),c(".macro-auto-pagination li a").off("click.taskreport.pagination"))});var k=a.closest(".table-wrapper");a=new g({$wrapper:k,table:a[0],sortReverseSortDefault:b.reverseSort,sortColumnDefault:g.getColumnNameFromSortBy(b.sortColumn),
reportParametersDefault:b,pageIndex:0,pageSize:f,pageTotal:d,adaptive:h,pageLimit:m,templates:l,columns:b.columns,onRenderEmptyTable:function(){Confluence.InlineTasks.TasksReport.renderBlankExperiences(k,b.status)},analyticEventKey:"confluence-spaces.tasks.report.sorted",restUrlPagination:"/rest/inlinetasks/1/task-report/",ajaxUrl:Confluence.getContextPath()+"/rest/inlinetasks/1/task-report/"});a.init();(h||1<d)&&a.renderPagination()})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-help-tips:common', location = 'js/help-tip.js' */
(function(f){function b(){return false}function d(){return true}var l=0,g=new Date().getTime();var k=AJS.contextPath()+"/rest/helptips/1.0/tips";function h(n){n=""+(n||"");return n.replace(/\./g,"-")}function j(o,n){if(AJS.EventQueue&&n&&n.attributes.id){var q={};var p=h(n.attributes.id);q.name="helptips."+p+"."+o;q.properties={};AJS.EventQueue.push(q)}}function c(){return"c"+g+(l++)}var a={dismissedTipIds:[],loaded:f.Deferred(),url:function(){return k},sync:function(o,n){o||(o="get");n||(n=null);return f.ajax(this.url(),{type:o,context:this,dataType:"json",contentType:"application/json",data:n&&JSON.stringify(n),processData:false}).promise()},fetch:function(){var n=this.sync();n.done(function(o){f.merge(this.dismissedTipIds,o);this.loaded.resolve()});return n.promise()},show:function(n){this.loaded.done(n)},dismiss:function(n){var o=n.attributes.id;if(!o){n._dismissed=true}else{this.dismissedTipIds.push(o);this.sync("post",{id:o})}},undismiss:function(n){var o=n.attributes.id;if(!o){n._dismissed=false}else{this.dismissedTipIds.splice(f.inArray(o,this.dismissedTipIds),1);this.sync("delete",{id:o})}},isDismissed:function(n){var o=n.attributes.id;return(o)?f.inArray(o,this.dismissedTipIds)>=0:n._dismissed}};var e=AJS.HelpTip=function(n){var o;this.attributes=f.extend({},n);this.attributes.id||(this.attributes.id=false);if(this.attributes.body){this.attributes.bodyHtml=this.attributes.body;delete this.attributes.body}this.cid=c();o=this.attributes.anchor;delete this.attributes.anchor;this.view=o?new i(this,o):new m(this);this.view.$el.addClass("aui-help-tip")};AJS.HelpTip.Manager=a;f.extend(e.prototype,{show:function(){var n=this;AJS.HelpTip.Manager.show(function(){if(!n.isDismissed()){if(AJS.Popups&&AJS.Popups.DisplayController){AJS.Popups.DisplayController.request({name:n.id,weight:1000,show:function(){n.view.show()}})}else{n.view.show()}j("shown",n)}})},dismiss:function(){var n=h(arguments[0]||"programmatically");this.view.dismiss();if(!this.isDismissed()){AJS.HelpTip.Manager.dismiss(this);j("dismissed."+n,this)}},isVisible:function(){return this.view.$el.is(":visible")},isDismissed:function(){return AJS.HelpTip.Manager.isDismissed(this)}});var i=function(o,n){this.initialize(o,n)};f.extend(i.prototype,{initialize:function(r,q){var o=this;var s=q.prop("ownerDocument");var p=s!=window.document;if(p){var n=f("iframe").filter(function(){return this.contentDocument==s});n.contents().scroll(function(){var v=f(this).contents().find("body").scrollTop();var w=n.offset().top;var u=o.popup.data("offset-top");var t=o.popup.find(".arrow").height();o.popup.css("top",u-v);o.popup.toggle(v<=u-w-t&&v+n.height()-t+w-o.popup.height()>=u)})}this.model=r;this.beforeHide=b;this.dismissButton=f(AJS.Templates.HelpTip.tipDismiss());this.dismissButton.click(function(t){r.dismiss("close-button");t.preventDefault()});this.popup=AJS.InlineDialog(q,r.cid,function(v,u,t){v.html(AJS.Templates.HelpTip.tipContent(r.attributes));v.find(".helptip-body").after(o.dismissButton);v.unbind("mouseover mouseout");v.find(".helptip-link").click(function(){j("learn-more.clicked",r)});t()},{container:"body",noBind:true,preHideCallback:function(){return o.beforeHide()},calculatePositions:function(t,y,z,x){var w=AJS.InlineDialog.opts.calculatePositions(t,y,z,x);if(p){var v=t.find(".arrow").height();var u=n.contents().find("body").scrollTop();w.popupCss.top=n.offset().top+(q.offset().top-u)+q.height()+v;w.popupCss.left=q.offset().left;n.contents().scroll()}t.data("offset-top",w.popupCss.top);return w}});this.popup.refresh();this._popupHide=this.popup.hide;this.popup.hide=b;this.$el=f(this.popup[0]);AJS.$(document).bind("showLayer",function(v,u,t){if(u==="inlineDialog"&&t.id===r.cid){AJS.InlineDialog.current=null;AJS.$(document.body).unbind("click."+r.cid+".inline-dialog-check");t._validateClickToClose=b;t.hide=b}})},show:function(){var n=this.popup,o=function(p){if(!n.has(p.target).length){n.shadow.remove();n.remove()}};n.show()},dismiss:function(){this.beforeHide=d;this._popupHide()}});var m=function(n){this.initialize(n)};f.extend(m.prototype,{initialize:function(){this.$el=f("<div />")},show:f.noop,dismiss:f.noop});if(AJS.Meta.get("remote-user")){a.fetch()}})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-help-tips:common', location = 'templates/help-tip.soy' */
// This file was automatically generated from help-tip.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace AJS.Templates.HelpTip.
 */

if (typeof AJS == 'undefined') { var AJS = {}; }
if (typeof AJS.Templates == 'undefined') { AJS.Templates = {}; }
if (typeof AJS.Templates.HelpTip == 'undefined') { AJS.Templates.HelpTip = {}; }


AJS.Templates.HelpTip.tipContent = function(opt_data, opt_ignored) {
  return ((opt_data.title) ? '<h4 class="helptip-title">' + soy.$$escapeHtml(opt_data.title) + '</h4>' : '') + '<div class="helptip-body">' + soy.$$filterNoAutoescape(opt_data.bodyHtml) + '</div>' + ((opt_data.url) ? '<a class="helptip-link" href="' + soy.$$escapeHtml(opt_data.url) + '" target="_blank">' + soy.$$escapeHtml('Learn more') + '</a>' : '');
};
if (goog.DEBUG) {
  AJS.Templates.HelpTip.tipContent.soyTemplateName = 'AJS.Templates.HelpTip.tipContent';
}


AJS.Templates.HelpTip.tipDismiss = function(opt_data, opt_ignored) {
  return '<button class="helptip-dismiss aui-button" type="button">' + soy.$$escapeHtml('Dismiss') + '</button>';
};
if (goog.DEBUG) {
  AJS.Templates.HelpTip.tipDismiss.soyTemplateName = 'AJS.Templates.HelpTip.tipDismiss';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-tasks:tasks-feature-discovery-resources', location = 'js/tasks-discovery.js' */
(function(a){a(function(){if(0<a("meta[name\x3dshow-task-feature-discovery-flag]").length&&Confluence.FeatureDiscovery.forPlugin("com.atlassian.confluence.plugins.confluence-jira-metadata").shouldShow("inline-tasks-flag")&&AJS.HelpTip){var b={id:null,body:Confluence.InlineTasks.Report.Templates.tasksFeatureDiscoveryNotification(),anchor:a("#user-menu-link")};b=new AJS.HelpTip(b);AJS.trigger("analyticsEvent",{name:"confluence-spaces.tasks.feature.discovery.shown"});b.show();Confluence.FeatureDiscovery.forPlugin("com.atlassian.confluence.plugins.confluence-jira-metadata").markDiscovered("inline-tasks-flag")}})})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.drag-and-drop:drag-and-drop-for-view-content', location = 'js/viewcontent-init.js' */
require(["jquery","window","wrm/require"],function(a,c,d){function b(){return new Promise(function(e,a){d("wrc!com.atlassian.confluence.plugins.drag-and-drop:default-drop-handler").then(function(){e(require("confluence-drag-and-drop/default-drop-handler"))},a)})}a(document).one("dragenter",function(){b().then(function(a){a.initialise()})});a(c).one("load",b)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-comments:util-resource', location = 'util/utils.js' */
define("confluence/ic/util/utils",["jquery","underscore","ajs","backbone","exports"],function(e,g,d,w,c){function x(a){return 0<a.filter(function(){return 0<e(this).text().length}).length}function m(a,b,c,d){null!=b&&(b.isResolved()?(new c({collection:a,focusCommentId:b.get("id")})).render():b.isDangling()||w.trigger("ic:view",b,"permalink",{isReplyComment:d}))}function p(a){if("number"===typeof a.selectionStart){var b=a.value.length;a.selectionStart=b;a.selectionEnd=b}else a.focus(),a=a.createTextRange(),
a.collapse(!1),a.select()}function q(a){var b=e(a).closest("a");b.length||(b=e(a).find("a"));return b}function r(){return!!e(".ic-sidebar #wysiwygTextarea_ifr").length}function t(){void 0===h&&(h=!!u().length);return h}function v(){void 0===k&&(k=e("#content"));return k}function u(){void 0===l&&(l=e("#splitter-content"));return l}var y={INLINE_COMMENTS:d.DarkFeatures.isEnabled("confluence-inline-comments"),RESOLVED_INLINE_COMMENTS:d.DarkFeatures.isEnabled("confluence-inline-comments-resolved"),RICH_TEXT_EDITOR:d.DarkFeatures.isEnabled("confluence-inline-comments-rich-editor"),
DANGLING_COMMENT:d.DarkFeatures.isEnabled("confluence-inline-comments-dangling-comment")},z=["dateautocomplete","confluencemacrobrowser","propertypanel","jiraconnector","dfe"],A=["autoresize","confluenceleavecomment"],B=["code"],k,l,h;c.overlappedSelection=function(a,b){if(a&&b){var c=e(a.containingElement);c.is(".inline-comment-marker.valid")||(c=e("\x3cdiv/\x3e").append(a.html).find(".inline-comment-marker.valid"));if(0<c.length&&x(c))return a=c.first().data("ref"),b=b.findWhere({markerRef:a})}};
c.focusedCommentUrl=function(a){return d.contextPath()+"/pages/viewpage.action?pageId\x3d"+d.Meta.get("latest-page-id")+"\x26focusedCommentId\x3d"+a+"#comment-"+a};c.showFocusedComment=function(a,b,c){b=b.match(/(focusedCommentId|replyToComment)=(\d+)/);if(null!=b){var n="replyToComment"===b[1];b=parseInt(b[2],10);var f=a.findWhere({id:b});null!=f?m(a,f,c,n):e.ajax({url:d.contextPath()+"/rest/inlinecomments/1.0/comments/replies/"+b+"/commentId"}).done(function(b){f=a.findWhere({id:b});m(a,f,c,n)})}};
c.getAuthorAvatarUrl=function(){return d.Meta.get("current-user-avatar-url")?d.contextPath()+d.Meta.get("current-user-avatar-url"):d.Meta.get("static-resource-url-prefix")+"/images/icons/profilepics/anonymous.png"};c.moveCaretToEnd=function(a){p(a);window.setTimeout(function(){p(a)},1)};c.animateSidebar=function(a,b){var c=a.$wikiContent,d=e.Deferred();if(b){var f=e(".inline-comment-marker.active");c.addClass("ic-fade-in-animation");c.one("webkitAnimationEnd oanimationend msAnimationEnd animationend",
function(){e(this).removeClass("ic-fade-in-animation")});a.$el.addClass("ic-slide-in");a.$el.one("webkitAnimationEnd oanimationend msAnimationEnd animationend",function(){e(this).removeClass("ic-slide-in");f.addClass("ic-highlight-pulse");f.one("webkitAnimationEnd oanimationend msAnimationEnd animationend",function(){f.removeClass("ic-highlight-pulse")})});d.resolve()}else{c.bind("webkitAnimationEnd oanimationend msAnimationEnd animationend",function(b){"ic-fade-out-animation"==b.originalEvent.animationName?
(a.$wikiContent.removeClass("ic-fade-out-animation"),a.$wikiContent.css("opacity","0.5")):"ic-fade-in-animation"==b.originalEvent.animationName&&(a.$wikiContent.css("opacity","1"),a.$wikiContent.removeClass("ic-fade-in-animation"),a.$wikiContent.unbind("webkitAnimationEnd oanimationend msAnimationEnd animationend"))});c.addClass("ic-fade-out-animation");a.$el.addClass("ic-slide-out");var g=function(){a.$el.removeClass("ic-slide-out");a._emptySidebar();v().css("padding-right","0");c.addClass("ic-fade-in-animation");
c.css("position","static");d.resolve();a.$el.off("webkitAnimationEnd oanimationend msAnimationEnd animationend",g)};a.$el.on("webkitAnimationEnd oanimationend msAnimationEnd animationend",g)}return d.promise()};c.getDarkFeatures=function(){return g.clone(y)};c.getInlineLinks=q;c.removeInlineLinksDialog=function(a){a.each(function(){var a=q(this);a.length&&a.off("mousemove")})};c.isAnimationSupported=function(){return void 0!==document.body.style.animation||void 0!==document.body.style.webkitAnimation};
c.showHighlightContent=function(a){a.parents(".expand-content.expand-hidden").each(function(a){e(this).siblings(".expand-control").click()})};c.getUnsupportedRtePlugins=function(){return g.clone(z)};c.getSupportedRtePlugins=function(){return g.clone(A)};c.confirmProcess=function(a){var b=d.Rte&&d.Rte.getEditor();if(!0===a){if(r()&&b&&b.isDirty())return confirm("Your comment will be lost.")}else if(b&&b.isDirty()&&Confluence.Editor&&!Confluence.Editor.getContentType)return confirm("Your comment will be lost.");
return!0};c.getAuthorDisplayName=function(){return d.Meta.get("user-display-name")||d.Meta.get("current-user-fullname")};c.isKeyboardShortcutsEnable=function(){return d.Meta.get("use-keyboard-shortcuts")&&Confluence.KeyboardShortcuts&&Confluence.KeyboardShortcuts.enabled};c.addSidebarHeadingTooltip=function(a){a.$("button.ic-action-hide").tooltip({gravity:"se"});a.$("#ic-nav-next").tooltip({gravity:"s"});a.$("#ic-nav-previous").tooltip({gravity:"s"})};c.hasEditorInSidebar=r;c.recalculateContentHeight=
function(a){if(t())a.css("padding-bottom","20px");else{var b=a.height();a=a.offset().top+b;b=this.getPageContainer().offset().top;a-=b;this.getPageContainer().css({"min-height":a+"px"})}};c.resizeContentAfterLoadingImages=function(a){a=a.closest(".ic-display-comment-view");var b=this,c=e(".confluence-embedded-image, .confluence-embedded-file img",a),d=c.length,f=0;0<d&&c.off("load").one("load",function(){++f===d&&b.recalculateContentHeight(a)}).each(function(){this.complete&&e(this).load()})};c.isDocTheme=
t;c.getPageContainer=v;c.getSplitterContent=u;c.scrollToCommentAfterToggleSideBar=function(a,b){(!b||b.is(":visible"))&&a&&(this.isDocTheme()?this.getSplitterContent().scrollTop(b.offset().top-a):window.scrollTo(0,b.offset().top-a))};c.getEditorFormat=function(a){var b=e.Deferred();e.ajax({url:d.contextPath()+"/rest/api/content/"+a+"?expand\x3dbody.editor",type:"GET",dataType:"json",contentType:"application/json; charset\x3dutf-8"}).then(function(a){b.resolve(a.body.editor.value)}).fail(function(a,
c,d){b.reject(a,c,d)});return b.promise()};c.containUnsupportedMacro=function(a){var b=a.closest(".conf-macro");return!1===b.data("hasbody")||0<a.find('.conf-macro[data-hasbody\x3d"false"]').length||-1!=B.indexOf(b.data("macro-name"))};c.containInternalLink=function(a){return 0<a.closest(".user-mention").length||0<a.find(".user-mention").length};c.containUserMetion=function(a){return 0<a.closest("a[href^\x3d'/']:not('.user-mention')").length||0<a.find("a[href^\x3d'/']:not('.user-mention')").length};
c.getCurrentUserInfo=function(){var a={isDefault:!0,path:d.Meta.get("static-resource-url-prefix")+"/images/icons/profilepics/default.png"};"/images/icons/profilepics/default.png"!==d.Meta.get("current-user-avatar-url")&&(a={isDefault:!1,path:d.contextPath()+d.Meta.get("current-user-avatar-url")});var b=d.Meta.get("remote-user");return{userName:""==b?null:b,displayName:d.Meta.get("current-user-fullname"),profilePicture:a}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-comments:util-resource', location = 'util/text-highlighter.js' */
define("confluence/ic/util/text-highlighter",["jquery"],function(a){function c(){a.textHighlighter.createWrapper=function(b){return a("\x3cspan\x3e\x3c/span\x3e").addClass(b.highlightedClass)};this.$el=a("#content .wiki-content").first();0<this.$el.length&&this.$el.textHighlighter({highlightedClass:"ic-current-selection"})}c.prototype.highlight=function(b){if(0!==this.$el.length)return b=a(this.$el.getHighlighter().doHighlight(b)),this.$el.getHighlighter().serializeHighlights(b)};c.prototype.removeHighlight=
function(){if(0!==this.$el.length)return this.$el.getHighlighter().removeHighlights(),this};c.prototype.deserializeHighlights=function(b,a){if(0!==this.$el.length)return a='\x3cspan class\x3d"inline-comment-marker" data-ref\x3d"'+a+'"\x3e\x3c/span\x3e',this.$el.getHighlighter().deserializeHighlights(b,a)};return c});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-comments:models', location = '/model/comment.js' */
define("confluence/ic/model/comment","jquery underscore backbone ajs confluence/ic/util/utils confluence/ic/model/reply-collection".split(" "),function(c,d,e,b,f,g){return e.Model.extend({defaults:{authorDisplayName:f.getAuthorDisplayName(),authorUserName:b.Meta.get("remote-user"),authorAvatarUrl:b.contextPath()+b.Meta.get("current-user-avatar-url"),body:"",originalSelection:"",containerId:b.Meta.get("latest-page-id"),containerVersion:b.Meta.get("page-version"),parentCommentId:"0",lastFetchTime:c("meta[name\x3dconfluence-request-time]").attr("content"),
hasDeletePermission:!0,hasEditPermission:!0,hasResolvePermission:!0,resolveProperties:{resolved:!1,resolvedTime:0},serializedHighlights:"",deleted:!1},urlRoot:b.contextPath()+"/rest/inlinecomments/1.0/comments",initialize:function(a){a=a||{};this._setHighlights(a.markerRef);var h=this;this.replies=new g;this.replies.url=function(){return h.url()+"/replies"};f.getDarkFeatures().DANGLING_COMMENT&&b.Meta.get("page-id")===b.Meta.get("latest-page-id")&&this.isDangling()&&!this.isResolved()&&this.resolve(!0,
{wait:!0,dangling:!0,success:d.bind(function(){e.trigger("ic:open:dangled",this)},this),error:d.bind(function(a,b){e.trigger("ic:resolve:dangled:failed",this)},this)})},validate:function(){if(!this.get("body"))return!0},resolve:function(a,b){b=b||{};this.save({},d.extend(b,{url:this.urlRoot+"/"+this.get("id")+"/resolve/"+a+"/dangling/"+(!0===b.dangling)}))},isResolved:function(){return this.get("resolveProperties").resolved},isDangling:function(){return void 0===this.highlight},isDeleted:function(){return this.get("deleted")},
_setHighlights:function(a){a=void 0!==a?c("#content .wiki-content:first").find('.inline-comment-marker[data-ref\x3d"'+a+'"]'):c(".ic-current-selection");0!==a.length?(this.highlights=a,this.highlight=a.first()):this.highlight=this.highlights=void 0},destroy:function(a){a=a?d.clone(a):{};var b=this,c=a.success;a.success=function(k){c&&c(b,k,a)};(function(a,b){var c=b.error;b.error=function(d){c&&c(a,d,b);a.trigger("error",a,d,b)}})(this,a);return this.sync("delete",this,a)}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-comments:models', location = '/model/comment-collection.js' */
define("confluence/ic/model/comment-collection",["jquery","backbone","ajs","underscore","confluence/ic/model/comment"],function(f,g,d,c,h){return g.Collection.extend({model:h,initialize:function(){this.listenTo(this,"sort",this._removeCachedMarkers)},url:function(){var a=d.contextPath(),b=d.Meta.get("page-id");return a+"/rest/inlinecomments/1.0/comments?containerId\x3d"+b},getResolvedCount:function(){return this.getResolvedComments().length},getUnresolveCount:function(){return this.reject(function(a){return!0===
a.isResolved()&&!1===a.isDeleted()}).length},getResolvedComments:function(){return this.filter(function(a){return!0===a.isResolved()&&!1===a.isDeleted()})},getResolvedCommentsDesc:function(){return c.sortBy(this.getResolvedComments(),function(a){return 0-a.get("resolveProperties").resolvedTime})},getNextCommentOnPage:function(){return this._getCommentAtRelativeOffset(1)},getPrevCommentOnPage:function(){return this._getCommentAtRelativeOffset(-1)},_getCommentAtRelativeOffset:function(a){var b=this.getCommentsOnPage(),
e=this.findWhere({active:!0});if(void 0===e||1>=b.length)return null;var d=c.pluck(b,"id");e=c.indexOf(d,e.get("id"));return b[(e+a+b.length)%b.length]},getCommentsOnPage:function(){return this.filter(function(a){return!1===a.isResolved()&&!a.isDangling()&&!1===a.isDeleted()||!0===a.get("active")})},getCommentsOnPageCount:function(){return this.getCommentsOnPage().length},getActiveIndexWithinPageComments:function(){var a=c.pluck(this.getCommentsOnPage(),"id"),b=this.findWhere({active:!0});if(void 0===
b)return null;b=b.get("id");return c.indexOf(a,b)},comparator:function(a,b){void 0===this.markers&&(this.markers=f("#content .wiki-content:first").find(".inline-comment-marker"));return void 0===a.highlight?1:void 0===b.highlight?-1:this.markers.index(a.highlight)-this.markers.index(b.highlight)},_removeCachedMarkers:function(){this.markers=void 0}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-comments:models', location = '/model/reply.js' */
define("confluence/ic/model/reply",["jquery","backbone","ajs","confluence/ic/util/utils"],function(f,d,a,c){return d.Model.extend({defaults:{authorDisplayName:c.getAuthorDisplayName(),authorUserName:a.Meta.get("remote-user"),authorAvatarUrl:a.contextPath()+a.Meta.get("current-user-avatar-url"),body:"",commentId:"0",hasDeletePermission:!0,hasEditPermission:!!a.Meta.get("remote-user")},urlRoot:function(){return a.contextPath()+"/rest/inlinecomments/1.0/comments/"+this.get("commentId")+"/replies"},sync:function(e,
c,b){b=b||{};"create"===e&&(b.url=this.urlRoot()+"?containerId\x3d"+a.Meta.get("latest-page-id"));return d.sync.call(this,e,c,b)},validate:function(){if(!this.get("body"))return!0}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-comments:models', location = '/model/reply-collection.js' */
define("confluence/ic/model/reply-collection",["backbone","confluence/ic/model/reply"],function(a,b){return a.Collection.extend({model:b})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-comments:app', location = 'analytics/ic-analytics.js' */
define("confluence/ic/analytics",["ajs","underscore","backbone"],function(c,d,a){var b={};d.extend(b,a.Events);b.start=function(){!0!==this.running&&(this.running=!0,this.send=function(a,b){c.trigger("analytics",{name:a,data:b})},this.listenTo(a,"ic:highlight-panel-click",function(){this.send("confluence.highlight.actions.comment.inline")}),this.listenTo(a,"ic:view",function(a,b){this.send("confluence.comment.inline.view");"nav"===b&&this.send("confluence.comment.inline.view.nav");"permalink"===b&&
this.send("confluence.comment.inline.view.permalink")}),this.listenTo(a,"ic:overlap",function(){this.send("confluence.comment.inline.overlap")}),this.listenTo(a,"ic:edit",function(){this.send("confluence.comment.inline.edit")}),this.listenTo(a,"ic:persist",function(){this.send("confluence.comment.inline.create")}),this.listenTo(a,"ic:sidebar:close",function(){this.send("confluence.comment.inline.sidebar.close")}),this.listenTo(a,"ic:reply:persist",function(){this.send("confluence.comment.inline.reply")}),
this.listenTo(a,"ic:delete ic:reply:delete",function(){this.send("confluence.comment.inline.delete")}),this.listenTo(a,"ic:resolved",function(){this.send("confluence.comment.inline.resolved")}),this.listenTo(a,"ic:unresolved",function(){this.send("confluence.comment.inline.unresolved")}),this.listenTo(a,"ic:resolved:view",function(a){this.send("confluence.comment.inline.resolved.view",{total:a})}),this.listenTo(a,"ic:resolved:dismiss:recovery",function(){this.send("confluence.comment.inline.resolved.dismiss")}),
this.listenTo(a,"ic:resolved:show:recovery",function(){this.send("confluence.comment.inline.resolved.discovery")}),this.listenTo(a,"ic:open:dangled",function(a){a={commentId:a.get("id"),pageId:c.Meta.get("page-id")};this.send("confluence.comment.inline.open.dangled",a)}),this.listenTo(a,"ic:editor:load:fail",function(){var a={pageId:c.Meta.get("page-id")};this.send("confluence.comment.inline.editor.load.fail",a)}),this.listenTo(a,"ic:resolve:dangled:failed",function(a){a={commentId:a.get("id"),pageId:c.Meta.get("page-id")};
this.send("confluence.comment.inline.resolved.dangled.failed",a)}))};b.stop=function(){this.running=!1;this.stopListening()};return b});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-comments:app', location = '/app/app.js' */
define("confluence/ic/app/app","jquery backbone ajs confluence/ic/model/comment confluence/ic/model/comment-collection confluence/ic/analytics exports".split(" "),function(g,h,e,q,r,t,k){k.init=function(){function l(){c().done(function(a){a.afterSyncCommentCollection();a.displayPermalinkedComment(b)})}function c(){if(n){n=!1;if(f)return f.promise();f=g.Deferred();WRM.require("wrc!inline-comments-load-sidebar").then(function(){var a=require("confluence/ic/app/loader");a.init(b);f.resolve(a)})}m=!1;
return f.promise()}t.start();var n=!0,b=new r;g("body").on("click","#view-resolved-comments",function(a){a.preventDefault();c().done(function(a){a.createResolvedCommentListView()})});b.fetch({cache:!1}).done(function(){0<b.getCommentsOnPageCount()&&l();0<b.getResolvedCount()&&c().done(function(a){a.updateResolvedCommentCountInToolsMenu()})});-1!==window.location.search.indexOf("focusedCommentId")&&l();h.listenTo(h,"ic:resolved",function(){l()});g("#view-resolved-comments\x3espan").text("Resolved comments"+
e.format(" ({0})",b.getResolvedCount()));var k=function(a,p){var d=b.get(p),c=!1;void 0===d&&(d=new q({id:p}),c=!0);d.fetch({success:function(a){a._setHighlights(a.get("markerRef"));c?b.add(a):d.replies.isFetched=!1;h.trigger("ic:view",d,"quickreload")}})};e.bind("qr:show-new-thread",function(a,b){c().done(function(){k(a,b)})});e.bind("qr:add-new-highlight",function(a,b,d){c().done(function(){e.trigger("qr:add-new-highlight-text",[b,d])})});var f,m;Confluence&&Confluence.HighlightAction&&Confluence.HighlightAction.registerButtonHandler("com.atlassian.confluence.plugins.confluence-inline-comments:create-inline-comment",
{onClick:function(a){c().done(function(b){0==m&&(m=!0,b.loadSidebarOnClick(a))})},shouldDisplay:Confluence.HighlightAction.WORKING_AREA.MAINCONTENT_ONLY})}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-inline-comments:bootstrap', location = '/app/bootstrap.js' */
require(["jquery","ajs","confluence/ic/app/app"],function(b,a,c){a.Meta.get("page-id")&&a.Meta.get("page-id")===a.Meta.get("latest-page-id")&&b(c.init)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-view-file-macro:view-file-macro-amd-resources', location = '/js/amd/confluence-amd.js' */
define("confluence",function(){return Confluence});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-view-file-macro:view-file-macro-amd-resources', location = '/js/amd/tinymce-amd.js' */
define("tinymce",function(){return tinymce});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-view-file-macro:common', location = '/js/view-file-macro-utils.js' */
define("vfm/view-file-macro-utils",[],function(){return{DEFAULT_HEIGHT:"250",DEFAULT_HEIGHT_IN_COMMENT:"150",THUMBNAIL_STATUS_IN_PROGRESS:202,THUMBNAIL_STATUS_CONVERTED:200,THUMBNAIL_STATUS_ERROR:415,THUMBNAIL_STATUS_BUSY:429,THUMBNAIL_POLLING_PERIOD:1E3,THUMBNAIL_POLLING_BACKOFF_RATIO:1.25,MAP_NICE_TYPE_TO_TEXT:{"pdf document":"PDF","word document":"Document","excel spreadsheet":"Spreadsheet",
"powerpoint presentation":"Presentation","generic file":"File"},getFileNameFromUrl:function(a){if(!a)return"";var b=a.indexOf("#");b=0<=b?b:a.length;a=a.substring(0,b);b=a.indexOf("?");b=0<=b?b:a.length;a=a.substring(0,b);b=a.lastIndexOf("/");a=a.substring(b+1,a.length);return decodeURIComponent(a)},isSupportPointerEvents:function(){var a=document.createElement("x");a.style.cssText="pointer-events:auto";return"auto"===
a.style.pointerEvents},getParameterByName:function(a,b){var c=a.indexOf("#");0<=c&&(a=a.substring(0,c));return(a=(new RegExp(b+"\x3d([^\x26]*)","i")).exec(a))?decodeURIComponent(a[1]):null},addParamsToUrl:function(a,b){var c=-1===a.indexOf("?")?"?":"\x26";for(var e=Object.keys(b),d=0;d<e.length;d++){var f=e[d],g=b[f];0<d&&(c+="\x26");c+=f+"\x3d"+g}return a+c},getFileTypeTextFromNiceType:function(a){a=a?a.toLowerCase():"";return this.MAP_NICE_TYPE_TO_TEXT[a]?this.MAP_NICE_TYPE_TO_TEXT[a]:a}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-view-file-macro:common', location = '/js/services/conversion-service.js' */
define("vfm/services/conversion-service",["ajs","jquery"],function(d,f){return{postThumbnailConversionResults:function(a){var b=d.contextPath()+"/rest/documentConversion/latest/conversion/thumbnail/results",c=Object.keys(a);c=_.map(c,function(e){return{id:e,v:a[e].version}});return f.ajax({type:"POST",url:b,contentType:"application/json",data:JSON.stringify(c)})},getThumbnailUrl:function(a,b){return d.contextPath()+"/rest/documentConversion/latest/conversion/thumbnail/"+a+"/"+b}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.editor:file-types-utils-resources', location = 'utils/file-types-utils.js' */
define("confluence-editor/utils/file-types-utils",[],function(){var b={},c={};"code|css,htm,html,java,js,xhtml,xml|text/css,text/html,text/javascript,text/x-java-source,text/xml doc|doc,docx,dot|application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document image|bmp,gif,jpe,jpeg,jpg,png,tif,tiff|image/bmp,image/gif,image/jpeg,image/png,image/tiff pdf|pdf|application/pdf ppt|pot,pps,ppt,pptx|application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation txt|asc,diff,log,text,txt|text/plain video|avi,flv,m4a,m4v,mov,mp2,mp3,mp4,mpe,mpeg,mpega,mpg,mpga,qt,rv,wav,webm,wmv|audio/mp3,audio/mp4,audio/mpeg,audio/x-wav,video/avi,video/mp4,video/mpeg,video/quicktime,video/vnd.rn-realvideo,video/webm,video/x-flv,video/x-m4v,video/x-ms-wmv xls|xlb,xls,xlsx|application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet zip|ear,jar,war,zip|application/java-archive,application/zip".split(" ").forEach(function(a){var a=
a.split("|"),d="aui-iconfont-file-"+a[0],e=a[1].split(",");a[2].split(",").forEach(function(a){b[a]=d});e.forEach(function(a){c[a]=d})});return{getAUIIconFromMime:function(a){return b[a]||"aui-iconfont-file-generic"},getAUIIconFromExt:function(a){return c[a]||"aui-iconfont-file-generic"},isImage:function(a){return b[a]&&0===a.indexOf("image/")}}});require("confluence/module-exporter").exportModuleAsGlobal("confluence-editor/utils/file-types-utils","AJS.Confluence.FileTypesUtils");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-view-file-macro:view-file-macro-embedded-file-view-soy-resources', location = '/templates/embedded-file-view.soy' */
// This file was automatically generated from embedded-file-view.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.ViewFileMacro.Templates.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.ViewFileMacro == 'undefined') { Confluence.ViewFileMacro = {}; }
if (typeof Confluence.ViewFileMacro.Templates == 'undefined') { Confluence.ViewFileMacro.Templates = {}; }


Confluence.ViewFileMacro.Templates.embeddedFile = function(opt_data, opt_ignored) {
  return '<span class="confluence-embedded-file-wrapper"><a class="confluence-embedded-file" href="' + soy.$$escapeHtml(opt_data.fileSrc) + '" data-nice-type="' + soy.$$escapeHtml(opt_data.niceType) + '" data-file-src="' + soy.$$escapeHtml(opt_data.fileSrc) + '" data-linked-resource-id="' + soy.$$escapeHtml(opt_data.attachmentId) + '" data-linked-resource-type="attachment" data-linked-resource-container-id="' + soy.$$escapeHtml(opt_data.containerId) + '" data-linked-resource-default-alias="' + soy.$$escapeHtml(opt_data.fileName) + '" data-mime-type="' + soy.$$escapeHtml(opt_data.mimeType) + '" data-has-thumbnail="' + ((opt_data.hasThumbnail) ? 'true' : 'false') + '" data-linked-resource-version="' + soy.$$escapeHtml(opt_data.attachmentVersion) + '" data-can-edit="' + soy.$$escapeHtml(opt_data.canEdit) + '" aria-label="' + soy.$$escapeHtml(opt_data.fileName) + '" draggable="false"' + ((opt_data.unresolvedCommentCount && opt_data.unresolvedCommentCount >= 0) ? 'data-unresolved-comment-count=' + soy.$$escapeHtml(opt_data.unresolvedCommentCount) : '') + '><img src="' + soy.$$escapeHtml(opt_data.placeholderSrc) + '" height="' + soy.$$escapeHtml(opt_data.height) + '" alt="" draggable="false" />' + ((! opt_data.hasThumbnail) ? '<span class="title">' + soy.$$escapeHtml(opt_data.fileName) + '</span>' : '') + '</a>' + ((opt_data.canEdit) ? Confluence.ViewFileMacro.Templates.companionEditButtonPlaceholder(opt_data) : '') + '</span>';
};
if (goog.DEBUG) {
  Confluence.ViewFileMacro.Templates.embeddedFile.soyTemplateName = 'Confluence.ViewFileMacro.Templates.embeddedFile';
}


Confluence.ViewFileMacro.Templates.embeddedUnknownFile = function(opt_data, opt_ignored) {
  return '<span class="confluence-embedded-file-wrapper"><img class="confluence-embedded-file unknown-attachment" src="' + soy.$$escapeHtml(opt_data.placeholderSrc) + '" draggable="false" /></span>';
};
if (goog.DEBUG) {
  Confluence.ViewFileMacro.Templates.embeddedUnknownFile.soyTemplateName = 'Confluence.ViewFileMacro.Templates.embeddedUnknownFile';
}


Confluence.ViewFileMacro.Templates.overlayEmbeddedFile = function(opt_data, opt_ignored) {
  return '<span class="overlay"></span>';
};
if (goog.DEBUG) {
  Confluence.ViewFileMacro.Templates.overlayEmbeddedFile.soyTemplateName = 'Confluence.ViewFileMacro.Templates.overlayEmbeddedFile';
}


Confluence.ViewFileMacro.Templates.overlayEmbeddedFileCommentCount = function(opt_data, opt_ignored) {
  return '<span class="comment-count-overlay"><span class="content">' + soy.$$escapeHtml(opt_data.commentCountRep) + '</span></span>';
};
if (goog.DEBUG) {
  Confluence.ViewFileMacro.Templates.overlayEmbeddedFileCommentCount.soyTemplateName = 'Confluence.ViewFileMacro.Templates.overlayEmbeddedFileCommentCount';
}


Confluence.ViewFileMacro.Templates.overlayEmbeddedFileFileTypeDesc = function(opt_data, opt_ignored) {
  return '<span class="file-type-desc-overlay"><i class="aui-icon aui-icon-small ' + soy.$$escapeHtml(opt_data.iconClass) + '"></i><span class="content"> ' + soy.$$escapeHtml(opt_data.fileType) + '</span></span>';
};
if (goog.DEBUG) {
  Confluence.ViewFileMacro.Templates.overlayEmbeddedFileFileTypeDesc.soyTemplateName = 'Confluence.ViewFileMacro.Templates.overlayEmbeddedFileFileTypeDesc';
}


Confluence.ViewFileMacro.Templates.companionEditButtonPlaceholder = function(opt_data, opt_ignored) {
  return '<span class="companion-edit-button-placeholder edit-button-overlay" data-linked-resource-container-id="' + soy.$$escapeHtml(opt_data.containerId) + '" data-linked-resource-id="' + soy.$$escapeHtml(opt_data.attachmentId) + '" data-template-name="companionEditIcon" data-source-location="embedded-attachment"></span>';
};
if (goog.DEBUG) {
  Confluence.ViewFileMacro.Templates.companionEditButtonPlaceholder.soyTemplateName = 'Confluence.ViewFileMacro.Templates.companionEditButtonPlaceholder';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-view-file-macro:view-file-macro-embedded-file-view-resources', location = '/js/services/file-service.js' */
define("vfm/services/file-service",["ajs","jquery"],function(b,c){return{getCommentCount:function(d,e){var a="/rest/files/1.0/files/content/{0}/commentCount?attachmentId\x3d{1}";a=b.contextPath()+b.format(a,d,e);return c.get(a)}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-view-file-macro:view-file-macro-embedded-file-view-resources', location = '/js/components/embedded-file-view.js' */
define("vfm/components/embedded-file-view",["jquery","backbone","ajs","confluence/legacy","vfm/view-file-macro-utils"],function(f,q,k,g,l){return{render:function(b){b=f(b.el);var a=b.find(".confluence-embedded-image, .confluence-embedded-file");if(a.hasClass("unknown-attachment")||a.attr("src")&&0<=a.attr("src").indexOf("/confluence/plugins/servlet/confluence/placeholder/unknown-attachment"))return this;var e="",h=a.hasClass("confluence-embedded-image"),c=b.parent().is("a"),m="true"===a.attr("data-has-thumbnail");
if(h)var d="image/png";else d=a.attr("data-mime-type"),e="null"!==a.attr("data-nice-type")?a.attr("data-nice-type"):"generic file";if(c)c="";else{c="";var n=a.attr("data-linked-resource-container-id"),p=a.attr("data-linked-resource-id");n&&p&&(a=a.attr("data-unresolved-comment-count"),a=parseInt(a,10),a=f.isNumeric(a)?a:0,a=9<a?"9+":a+"","0"!==a&&(c=g.ViewFileMacro.Templates.overlayEmbeddedFileCommentCount({commentCountRep:a})))}d=!h&&m?g.ViewFileMacro.Templates.overlayEmbeddedFileFileTypeDesc({fileType:l.getFileTypeTextFromNiceType(e),
iconClass:k.Confluence.FileTypesUtils.getAUIIconFromMime(d)}):"";if(c||d)e=g.ViewFileMacro.Templates.overlayEmbeddedFile(),b.append(f(e).append(c).append(d)),d&&b.addClass("has-comment-overlay")}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-view-file-macro:view-file-macro-resources', location = '/js/vfm.js' */
require(["jquery","ajs","vfm/components/embedded-file-view"],function(a,b,c){a(document).on("click",".confluence-embedded-file.unknown-attachment",function(d){d.preventDefault()});b.toInit(function(){a(".confluence-embedded-file-wrapper").each(function(){c.render({el:this})})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.editor:editor-core-resources', location = 'loader/profiles.js' */
define("confluence-editor/profiles",["jquery","ajs"],function(d,c){return{createProfileForCommentEditor:function(){return{plugins:"searchreplace confluenceimagedialog autocompletemacro confluencemacrobrowser confluenceleavecomment confluencewatch autoresize".split(" ")}},createProfileForPageEditor:function(a){var b="searchreplace confluencedrafts confluenceimagedialog autocompletemacro confluencemacrobrowser flextofullsize referrer".split(" ");c.Meta.getBoolean("shared-drafts")&&c.DarkFeatures.isEnabled("unpublished-changes-lozenge")&&
b.push("unpublishedchanges");a&&a.versionComment&&b.push("confluenceversioncomment");a&&a.notifyWatchers&&b.push("confluencenotifywatchers");return{plugins:b}},createProfileForTemplateEditor:function(){return{plugins:"searchreplace confluenceimagedialog autocompletemacro confluencemacrobrowser confluenceleavetemplate flextofullsize confluencetemplateeditor".split(" ")}}}});require("confluence/module-exporter").exportModuleAsGlobal("confluence-editor/profiles","AJS.Confluence.Editor._Profiles");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-edit-util', location = 'jscripts/util.js' */
define("confluence-quick-edit/util",["window","ajs"],function(c,d){return{generateUUID:function(){var a=function(){return Math.floor(65536*(1+Math.random())).toString(16).substring(1)};return a()+a()+"-"+a()+"-"+a()+"-"+a()+"-"+a()+a()+a()},getBaseUrl:function(){var a=c.location.protocol.replace(/:$/,"")+"://"+c.location.host+"/"+c.location.pathname.replace(/^\//,""),b=c.location.search.replace(/^\?/,""),b=b.replace(/&?focusedCommentId=\d+/,""),b=b.replace(/^&/,"");return{url:a,search:b,addQueryParam:function(a,
b){this.search=this.search?this.search+"&"+a+"="+b:a+"="+b},toString:function(){return this.url+"?"+this.search}}},timeoutDeferred:function(a,b,c){"function"!==typeof b.reject&&d.log("WARNING: invalid, not rejectable object passed to AJS.Confluence.QuickEdit.Util.timeoutDeferred. You should use a Deferred object");setTimeout(function(){"pending"===b.state()&&(d.logError("Timeout: "+a),b.reject("timeout"))},c);return b}}});
require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/util","AJS.Confluence.QuickEdit.Util");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:querystring', location = '/includes/js/api/querystring.js' */
define("confluence/api/querystring",[],function(){return{stringify:function(a){var b="",c;for(c in a)for(var d=0;d<a[c].length;d++)b+="&"+c,a[c][d]&&(b+="="+a[c][d]);return b.substring(1)},parse:function(a){var b={};if(a){"?"===a.substr(0,1)&&(a=a.substr(1));for(var a=a.split("&"),c=0;c<a.length;c++){var d=a[c].split("=");b[d[0]]||(b[d[0]]=[]);b[d[0]].push(a[c].substring(d[0].length+1))}}return b}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:url', location = '/includes/js/api/url.js' */
define("confluence/api/url",["confluence/api/querystring","jquery"],function(d,e){var g=/([^?|#]+)[?]?([^#]*)[#]?(.*)/,f=["source","urlPath","queryParams","anchor"];return{parse:function(a){var b={};if(a=g.exec(a)){for(var c=0;c<f.length;c++)b[f[c]]=a[c];b.queryParams=d.parse(b.queryParams)}return b},format:function(a){return e.isEmptyObject(a)?"":(!a.urlPath?"":a.urlPath)+(e.isEmptyObject(a.queryParams)?"":"?"+d.stringify(a.queryParams))+(!a.anchor?"":"#"+a.anchor)}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:analytics-support', location = '/includes/js/analytics-support.js' */
define("confluence/analytics-support","jquery ajs confluence/meta window document confluence/api/url".split(" "),function(j,o,l,k,m,h){var b={},n;b.setAnalyticsSource=function(a,d){if(typeof n==="undefined"){var c=j._data(k,"events");n=c&&c.analytics&&c.analytics.length>0}n&&a.attr("href",function(a,c){var b=encodeURIComponent(d),g=h.parse(c);if(!j.isEmptyObject(g))g.queryParams.src=[b];return h.format(g)})};b.srcRemovedUrl=function(a){a=h.parse(a);delete a.queryParams.src;for(var d=Object.keys(a.queryParams),
c=0;c<d.length;c++){var b=d[c],f=b.split(".");f.length===3&&f[0]==="src"&&delete a.queryParams[b];b==="jwt"&&delete a.queryParams[b]}return h.format(a)};b.srcParamValues=function(a){return(a=h.parse(a).queryParams)&&a.src?a.src:[]};b.srcAttrParamValues=function(a){for(var a=h.parse(a).queryParams,b={},c=Object.keys(a),e=0;e<c.length;e++){var f=c[e],i=f.split(".");if(i.length===3&&i[0]==="src"){var g=i[1],i=i[2];b[g]=b[g]||{};b[g][i]=decodeURIComponent(a[f][0])}}return b};b.extractAnalyticsData=function(a){for(var d=
[],c=b.srcParamValues(a),a=b.srcAttrParamValues(a),e=0;e<c.length;e++){var f=c[e];d.push({src:f,attr:a[f]||{}})}return d};b.publish=function(a,b){o.trigger("analytics",{name:a,data:b||{}})};b.init=function(){var a=b.extractAnalyticsData(m.URL),d={userKey:l.get("remote-user-key"),pageID:l.get("page-id"),draftID:l.get("draft-id")};if(a.length>0){for(var c=0;c<a.length;c++){var e=a[c],f=j.extend({},d,e.attr);b.publish("confluence.viewpage.src."+e.src,f)}if(k.history&&k.history.replaceState){a=b.srcRemovedUrl(m.URL);
m.URL!==a&&k.history.replaceState(null,"",a)}}else b.publish("confluence.viewpage.src.empty",d)};return b});require("confluence/module-exporter").exportModuleAsGlobal("confluence/analytics-support","AJS.Confluence.Analytics",function(j){require("ajs").toInit(j.init)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-edit-general', location = 'jscripts/captcha-manager.js' */
define("confluence-quick-edit/captcha-manager",["jquery","ajs"],function(d,e){return function(f){function a(a){return d(f||"body").find(a)}return{refreshCaptcha:function(){var b=a("img.captcha-image");if(0<b.length){var c=Math.random();b.attr("src",e.contextPath()+"/jcaptcha?id="+c);a('input[name="captchaId"]').val(c);a('input[name="captchaResponse"]').val("")}},getCaptchaData:function(){return{id:a('input[name="captchaId"]').val(),response:a('input[name="captchaResponse"]').val()}}}}});
require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/captcha-manager","AJS.Confluence.QuickEditCaptchaManager");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-edit-general', location = 'jscripts/quick-edit.js' */
define("confluence-quick-edit/quick-edit","ajs confluence-editor-loader/block-and-buffer-keys confluence/legacy confluence/templates confluence/meta jquery window document confluence-editor-loader/editor-loader confluence/api/event confluence/api/logger confluence-quick-edit/captcha-manager confluence-quick-edit/util wrm".split(" "),function(i,u,e,r,j,a,v,w,o,c,g,k,x,y){function z(){var b=new a.Deferred;o.load(function(){setTimeout(function(){b.resolve()},0)},function(){b.reject()});return b}var s=
{enableShortcut:function(){a("#editPageLink").addClass("full-load")},disableShortcut:function(){a("#editPageLink").removeClass("full-load")}},l=[],q={loadingContentTimeout:4E3,register:function(b){l.push(b)},disableHandlers:function(){a.each(l,function(b,a){return a.disable()})},enableHandlers:function(){a.each(l,function(b,a){return a.enable()})},SaveBarBinder:{bind:function(b,a){b&&e.Editor.addSaveHandler(b);a&&e.Editor.addCancelHandler(a)}},activateEditor:function(b){function e(){function f(b){var c=
new a.Deferred;y.require(b).done(function(b){c.resolve(b)}).fail(function(b){c.reject(b)});return c}var l,m=new a.Deferred;if(j.get("access-mode")==="READ_ONLY"){g.logError("activateEditor could not be initialised: Read-only mode is enabled");return m.reject("READ_ONLY")}if(i.Rte&&i.Rte.getEditor()){g.debug("there is already an editor open");return m.reject("EDITOR_OPEN")}if(!b.$container||!b.$form){g.logError("activateEditor could not be initialised: bad arguments",b);return m.reject("BAD_ARGS")}l=
u.block(a(w));b.preActivate&&b.preActivate();s.disableShortcut();var t=b.timeoutResources||o.loadingTimeout,k=q.loadingContentTimeout,n=x.timeoutDeferred;a.when(n("resources",z(),t),n("fetch content",b.fetchContent||a.Deferred().resolve(),k),n("additional resources",b.additionalResources?f(b.additionalResources):a.Deferred().resolve(),t)).done(function(e,f){var d={$container:b.$container,content:f,$form:b.$form,replayBufferedKeys:l};b.preInitialise&&b.preInitialise(d);var h=o.getPreloadContainer();
a(".quick-comment-prompt",d.$container).hide();a(".quick-comment-body",d.$container).addClass("comment-body");if(d.content&&d.content.title){var p=d.content.title;h.find("#content-title").val(p)}d.$form.append(h.children());h.show();a("#editor-precursor").hide();a("#rte-savebar").find(".toolbar-split-left").hide();if(j.get("content-type")==="comment"){a("#pluggable-status").hide();var h=r.Editor.Page.cancelButton({contentType:j.get("content-type"),sharedDraftsEnabled:j.getBoolean("shared-drafts")}),
p=a("#rte-button-cancel"),k=p.parent(".rte-toolbar-group-done");if(k.length){k.remove();a("#rte-button-discard").remove()}else p.remove();a("#rte-savebar").find(".toolbar-split-right").append(h);h=r.Editor.Page.previewButton({});a("#rte-button-ellipsis").parent().remove();a("#rte-savebar").find(".toolbar-group-preview").empty().append(h)}var n=function(){d.editor=i.Rte.getEditor();d.$container.find(".quick-comment-loading-container").hide();d.$form.show();d.$form.addClass("fadeIn");var a=d.editor,
e=d.content?d.content.editorContent:"",f=d.replayBufferedKeys;if(e){g.debug("Initial Editor Content from quick edit: ",e);a.setContent(e);a.startContent=a.getContent({format:"raw",no_events:1});a.undoManager.clear()}f()&&a.undoManager.add();c.trigger("quickedit.success");c.trigger("quickedit.visible");c.trigger("add-bindings.keyboardshortcuts");c.trigger("active.dynamic.rte");b.postInitialise&&b.postInitialise(d);q.SaveBarBinder.bind(b.saveHandler,b.cancelHandler);c.trigger("rte-quick-edit-ready");
a=j.get("content-type");j.get("collaborative-content")&&(a==="page"||a==="blogpost")&&c.trigger("rte-collab-editor-loaded");c.unbind("rte-ready",n);m.resolve()};c.bind("rte-ready",n);c.bind("rte-destroyed",b.postDeactivate||function(){});i.Rte.BootstrapManager.initialise({plugins:b.plugins,toolbar:b.toolbar,excludePlugins:b.excludePlugins,isQuickEdit:true,onInitialise:b.onInitialise})}).fail(function(a,e){m.reject(a,e);g.logError("Error loading page quick edit. Falling back to normal edit URL...");
c.trigger("analytics",{name:"rte.quick-edit.activate-editor.failed"});if(b.fallbackUrl){g.log("This parameter is deprecated. To be removed in the next major version (5.8 or 6.0). Please use the promise returned to bind custom action if the editor fails to load instead.");v.location=b.fallbackUrl}});return m.promise()}if(b.closeAnyExistingEditor&&i.Rte&&i.Rte.getEditor()){var f=new a.Deferred;this.deactivateEditor().done(function(){e().done(function(){f.resolve()}).fail(function(a){f.reject(a)})}).fail(function(){g.debug("Could not deactivate current editor.");
f.reject("Could not deactivate current editor.")});return f.promise()}return e()},deactivateEditor:function(){require("tinymce").majorVersion>=4?require("tinymce").execCommand("mceRemoveEditor",true,"wysiwygTextarea"):require("tinymce").execCommand("mceRemoveControl",true,"wysiwygTextarea");e.Editor.UI.toggleSavebarBusy(false);var b=o.getPreloadContainer().empty();a(".editor-container").remove();a("#editor-precursor").remove();a("#anonymous-warning").remove();a(".quick-comment-prompt").show();a(".bottom-comment-panels").show();
a("#editor-notification-container").empty();a(".action-reply-comment").removeAttr("reply-disabled");s.enableShortcut();e.Editor.removeAllSaveHandlers();e.Editor.removeAllCancelHandlers();return o.getEditorPreloadMarkup().done(function(a){b.append(a);b.hide();(new k(b)).refreshCaptcha();c.trigger("rte-destroyed");c.unbind("rte-destroyed")})}};return q});require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/quick-edit","AJS.Confluence.QuickEdit");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-edit-general', location = 'jscripts/handlers/view.js' */
define("confluence-quick-edit/handlers/view",["jquery","ajs","window"],function(c,d,e){return function(){var a;sessionStorage.viewPort&&(a=JSON.parse(sessionStorage.viewPort));if(a&&a.pageId==d.params.pageId){var b;b=c("#main-content");var f=c("#header");b=-1!==a.blockIndex?b.children().first().children().eq(a.blockIndex).find(".innerCell").eq(a.columnIndex).children().eq(a.index):b.children().eq(a.index);e.scrollTo(0,b.offset().top+a.offset-f.outerHeight())}delete sessionStorage.viewPort}});
require("confluence/module-exporter").safeRequire("confluence-quick-edit/handlers/view",function(c){require("ajs").toInit(c)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-edit-general', location = 'jscripts/handlers/shortcut.js' */
define("confluence-quick-edit/handlers/shortcut",["ajs"],function(a){var d=!1;a.bind("initialize.keyboardshortcuts add-bindings.keyboardshortcuts",function(){d=!0});a.bind("remove-bindings.keyboardshortcuts",function(){d=!1});return{createShortcut:function(f,g){function b(){c=c||a.whenIType(f).moveToAndClick(g)}function e(){c&&c.unbind();c=null}var c;return{bind:function(){d&&b();a.bind("initialize.keyboardshortcuts",b);a.bind("add-bindings.keyboardshortcuts",b);a.bind("remove-bindings.keyboardshortcuts",
e)},unbind:function(){e();a.unbind("initialize.keyboardshortcuts",b);a.unbind("add-bindings.keyboardshortcuts",b);a.unbind("remove-bindings.keyboardshortcuts",e)}}}}});require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/handlers/shortcut","AJS.Confluence.QuickEdit.KeyboardShortcuts");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-edit-general', location = 'jscripts/handlers/page.js' */
define("confluence-quick-edit/handlers/page","jquery ajs confluence/legacy confluence/analytics-support wrm/context-path confluence/dark-features confluence/api/event confluence/api/logger confluence/meta confluence/aui-overrides window confluence/api/browser confluence-editor/editor/page-editor-message confluence/message-controller confluence-quick-edit/quick-edit confluence-quick-edit/handlers/shortcut".split(" "),function(b,i,l,p,s,I,f,o,g,A,m,B,C,n,t,D){function q(){var a=b("#editPageLink");a.find(".aui-icon").css("visibility",
"visible");a.parent().spinStop()}function u(){var a=g.get("content-type");return g.get("collaborative-content")&&(a==="page"||a==="blogpost")}function v(a,c,d){var e={pageId:d,blockIndex:-1,columnIndex:-1,index:-1,offset:0,hasBlock:function(){return this.blockIndex!==-1}},h=false,f=function(a){var b=a.offset();return b.top-8<=c&&b.top+a.height()>=c},g=function(a,b){var d=a.offset();if(f(a)){e.index=b;e.offset=c-d.top;h=true}};if(a.children().length===1&&a.children().first().hasClass("contentLayout2")){a.children().first().children().each(function(a){if(!e.hasBlock()&&
f(b(this)))e.blockIndex=a});if(e.hasBlock()){a=a.children().first().children().eq(e.blockIndex).find(".innerCell");a.each(function(a){if(e.columnIndex===-1){var d=b(this).children().length;if(d>0)if(d<2){if(b(this).children().first().height()>25)e.columnIndex=a}else e.columnIndex=a}});a.eq(e.columnIndex).children().each(function(a){h||g(b(this),a)})}}else a.children().each(function(a){h||g(b(this),a)});return h?e:null}function E(a){var c=require("tinymce");j.disable();A.metaToParams();var d=b("#main-content"),
e=b("#header"),h=b("#main-header"),e=m.pageYOffset+e.outerHeight()+h.outerHeight();k=v(d,e,i.params.pageId);f.trigger("quick-edit.viewport.saved");var w=function(){b(c.activeEditor.getWin().document).find("body#tinymce").addClass("page-edit");b("#content").css({paddingRight:0});f.unbind("quickedit.visible",w)};f.bind("quickedit.visible",w);d=a.$form;e=g.get("content-type")==="page"?"doeditpage":"doeditblogpost";e=s()+"/pages/"+e+".action?pageId="+l.getContentId();b(".ia-splitter-left").remove();try{b("#main").unwrap()}catch(n){}b("#rte").removeClass("editor-default").addClass("editor-fullheight");
a.$container.children().remove();b(".editor-container").children().eq(0).unwrap();d.attr({"class":"editor aui",action:e,name:"editpageform",id:"editpageform",style:""});a.$container.append(d);a.$container.removeClass("view").addClass("edit");b("body").addClass("contenteditor edit")}function F(a){require("tinymce");b("#editor-precursor").show();b("#rte-savebar").find(".toolbar-split-left").show();if(m.history.pushState){var c=b("#editPageLink").attr("href");if(c!==m.location.pathname+m.location.search){history.pushState({quickEdit:true},
"",c);f.trigger("rte-quick-edit-push-state",c)}}else{m.location.hash="editor";f.trigger("rte-quick-edit-push-hash")}c=a.content;if(c.permissions)for(var d in c.permissions)b("#"+d).attr("value",c.permissions[d]);b("#originalVersion").val(a.content.pageVersion);g.set("page-version",a.content.pageVersion);g.set("page-title",a.content.title);b('meta[name="page-version"]').attr("content",a.content.pageVersion);b('meta[name="ajs-page-version"]').attr("content",a.content.pageVersion);b("#syncRev").val(a.content.syncRev);
g.set("conf-revision",a.content.confRev);b('meta[name="ajs-conf-revision"]').attr("content",a.content.confRev);d=a.content.atlToken;g.set("atl-token",d);b('input[name="atl_token"]').val(d);f.trigger("analyticsEvent",{name:"quick-edit-success"});b("#navigation").remove();var e=new Date,h=function(b){if(e&&!(b.type==="keydown"&&[91,92,93,224,33,34,37,38,39,40,16,17,18,20,112,113,114,115,116,117,118,119,120,121,122,123].indexOf(b.keyCode)>-1)){b=new Date-e;e=null;p.publish("confluence.editor.transition.firstkeydown",
{delay:b});a.editor.off("keydown",h);a.editor.off("Change",h)}};a.editor.on("keydown",h);a.editor.on("Change",h);l.debugTimeEnd("confluence.editor")}function G(a){var c=function(){f.unbind("rte-ready",c);if(k){var d;d=b(a.getBody());d=k.hasBlock()?d.children().first().children().eq(k.blockIndex).find(".innerCell").eq(k.columnIndex).children().eq(k.index):d.children().eq(k.index);a.getWin().scrollTo(0,d.offset().top+k.offset);b("#main").css("visibility","visible")}};return function(){f.bind("rte-ready",
c)}}function x(a){f.trigger("rte-collaborative-content-ready",a)}function H(){var a=new b.Deferred;l.debugTime("confluence.editor.quick.fetchContent");var c=b.ajax({url:s()+"/rest/tinymce/1/content/"+l.getContentId()+".json",cache:false});c.success(function(b){g.get("edit-mode")&&g.get("edit-mode")!==b.editMode&&a.reject("edit mode change",c);l.debugTimeEnd("confluence.editor.quick.fetchContent");u()&&x(b);f.bind("synchrony-events-bound",function h(){x(b);f.unbind("synchrony-events-bound",h)});a.resolve(b)});
c.error(function(b){a.reject("error fetching content",b)});return a}function y(a,c){if(c)switch(c.status){case 405:q();n.showError(n.parseError(c),n.Location.FLAG);return;case 423:var d=JSON.parse(c.responseText).user;q();d={title:"Unable to edit",body:i.format("Collaborative editing is offline right now, and {0} is editing this page. Try again in a few minutes.",i.escapeHtml(d))};n.showError(d,n.Location.FLAG);return;case 412:q();C.handleMessage("collab.edit.user.limit.reached",{type:"warning",title:"You can\u0027t edit this page right now",
message:"\u003cp\u003eThis page is so popular, you\u0027ve reached the maximum number of simultaneous editors.\u003c/p\u003e\u003cp\u003eTry again when it\u0027s not so busy.\u003c/p\u003e",close:"manual"});p.publish("collab.edit.user.limit.reached",{browserName:z.friendlyName(),browserVersion:z.version(),pageId:g.get("page-id"),editMode:"quick",numEditors:g.get("max-number-editors")});return}m.location=b("#editPageLink").attr("href")}var r,k,z=new B(m.navigator.userAgent),j={editShortcut:D.createShortcut("e","#editPageLink"),activateEventHandler:function(a){if(!a.metaKey&&!a.shiftKey&&!a.ctrlKey&&!a.altKey&&!(a.which===2||a.which===
3)){a.preventDefault();if(r&&r.state()==="pending")o.debug("Editor is being activated. Ignoring handler...");else{r=j.activateEditor();a=b("#editPageLink");a.find(".aui-icon").css("visibility","hidden");a.spin();a=b("#draft-status-lozenge");a.text()!==""&&p.publish("confluence.drafts.referrer",{referrerPage:"view",lozengeType:a.text()})}}},enable:function(){if(b("body").is(".theme-default")){var a=b("#editPageLink");a.bind("click",j.activateEventHandler);a.removeClass("full-load");j.editShortcut.bind();
o.debug("QuickPageEdit enabled")}else o.debug("QuickPageEdit not enabled")},activateEditor:function(){l.debugTime("confluence.editor");u()&&f.trigger("rte-quick-edit-init");var a=b("#content").find(".quick-comment-form"),c=function(){var a=require("tinymce").activeEditor.getWin(),c=b(a.document).find("#tinymce");if(a=v(c,a.pageYOffset,i.params.pageId))sessionStorage.viewPort=JSON.stringify(a)};return t.activateEditor({fetchContent:H(),$container:b("#content"),$form:a.length?a:b('<form method="post"></form>'),
preInitialise:E,postInitialise:F,saveHandler:function(){c()},cancelHandler:function(){c()},plugins:l.Editor._Profiles.createProfileForPageEditor({versionComment:true,notifyWatchers:true}).plugins,onInitialise:function(a){a.on("Load",G(a));i.messages.setup()}}).fail(y)},disable:function(){o.debug("QuickPageEdit disabled.");j.editShortcut.unbind();b("#editPageLink").unbind("click",j.activateEventHandler)}};t.register(j);return{disable:j.disable,_objForTesting:{onFailActivateEditor:y}}});
require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/handlers/page","AJS.Confluence.QuickEdit.QuickEditPage");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-edit-general', location = 'jscripts/init.js' */
define("confluence-quick-edit/init",["ajs","jquery","window"],function(a,b,c){return function(){b("body").hasClass("page-gadget")?a.debug("QuickComment: editor preload is disabled"):b(c).load(function(){a.debug("QuickComment: instigated background loading of the comment editor.");a.Confluence.EditorLoader.load()});a.Confluence.QuickEdit.enableHandlers();a.trigger("rte-quick-edit-enable-handlers")}});
require("confluence/module-exporter").safeRequire("confluence-quick-edit/init",function(a){var b=require("ajs"),c=require("jquery");b.DarkFeatures.isEnabled("disable-quick-edit")?b.log("disable-quick-edit is turned on; run AJS.Confluence.EditorLoader.load() manually."):c(a)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-comment-initial', location = 'jscripts/handlers/comment.js' */
define("confluence-quick-edit/handlers/comment","jquery ajs confluence/legacy wrm/context-path confluence-editor-loader/editor-loader confluence/api/event aui/flag confluence/api/logger confluence-editor/editor/page-editor-message confluence/message-controller confluence/meta confluence-quick-edit/captcha-manager confluence-quick-edit/handlers/page confluence-quick-edit/util".split(" "),function(f,b,i,o,p,l,t,q,e,g,c,r,s,m){function n(a,h){var b=a.match(RegExp("[?&]"+h+"=(\\d+)"));return b&&b.length>
1?parseInt(b[1],10):0}f(function(){b.AppLinksInitialisationBinder=function(a){l.bind("init.rte",a)}});var j={timeout:8E3,showLoadingEditorErrorMessage:function(a){var h={};l.trigger("rte-quick-comment-loading-failed");if(a==="READ_ONLY"){h.title="This site is read-only";h.body="You can\u0027t make changes right now."}else{h.title="Error";h.body="Error loading the comment editor. Please refresh the page to try again."}g.showError(h,
g.Location.FLAG)},preInitialise:function(){s.disable();c.set("content-type","comment");c.set("draft-type","");b.params.contentType="comment";b.params.draftType="";c.set("use-inline-tasks","false");f("#editor-precursor").children().eq(0).hide();f("#pagelayout-menu").parent().hide();f("#page-layout-2-group").hide();f("#rte-button-tasklist").remove();f("#pluggable-status-container").remove();f("#rte-insert-tasklist").parent().remove()},postInitialise:function(a){b.Rte.editorFocus(a.editor)},delegatingSaveCommentHandler:function(a){return a.asyncRenderSafe?
j.ajaxSaveCommentHandler(a):j.reloadPageSaveCommentHandler(a)},reloadPageSaveCommentHandler:function(a){var b=m.getBaseUrl();b.addQueryParam("focusedCommentId",a.id);b.addQueryParam("refresh",+new Date);window.location.href=b.toString()+"#comment-"+a.id},ajaxSaveCommentHandler:function(a){var b={isDefault:true,path:c.get("static-resource-url-prefix")+"/images/icons/profilepics/default.png"};c.get("current-user-avatar-uri-reference")!==o()+"/images/icons/profilepics/default.png"&&(b={isDefault:false,
path:c.get("current-user-avatar-uri-reference")});var d=c.get("remote-user"),f={userName:d===""?null:d,displayName:c.get("current-user-fullname"),profilePicture:b};j.cancelComment().done(function(){i.CommentDisplayManager.addOrUpdateComment(f,a,true,false);l.trigger("page.commentAddedOrUpdated",{commentId:a.id})})},cancelHandler:function(){var a=require("confluence-quick-edit/quick-edit");b.Rte.Content.editorResetContentChanged();a.deactivateEditor()},createCommenterParam:function(a,b,d){return{userName:b||
c.get("remote-user")||null,displayName:d||c.get("user-display-name"),profilePicture:{isDefault:a.hasClass("defaultLogo"),path:a.attr("src")}}},createSaveHandler:function(a,h){var d=m.generateUUID();return function(c){var k=require("confluence/editor-notifications");c.preventDefault();if(b.Rte.Content.isEmpty()){k.notify("warning","Comment text is empty. Cannot create empty comments.");i.Editor.UI.toggleSavebarBusy(false)}else{var k=c=0,e=p.getEditorForm();if(e.is("form")){k=e.attr("action");c=n(k,"parentId");k=n(k,"commentId")}var g=
new r(e),e=function(a,b){h(a,b);g.refreshCaptcha()},j=f("#watchPage").is(":checked");k>0?i.Editor.CommentManager.updateComment(i.getContentId(),k,b.Rte.Content.getHtml(),j,g.getCaptchaData(),a,e):i.Editor.CommentManager.saveComment(i.getContentId(),c,b.Rte.Content.getHtml(),j,d,g.getCaptchaData(),a,e)}}},saveCommentErrorHandler:function(a,c){var d;i.Editor.UI.toggleSavebarBusy(false);if(a&&a.search(/captcha/i)!==-1){d="The typed word did not match the text in the picture.";e.closeMessages(["captcha-response-failed"]);
e.handleMessage("captcha-response-failed",{type:"error",message:d})}else if(a&&a.search(/Unsupported character found in content: (.{12})/i)!==-1){d=JSON.parse(a.replace(/error: {2}- /,"")).message.split("Unsupported character found in content: ")[1];d=b.format("We can\u0027\u0027t save your comment because the {0} character isn\u0027\u0027t supported by your database.\u003cbr\u003e\u003cbr\u003eRemove this character or use a Confluence symbol or emoticon instead. \u003ca href = \u0022https://confluence.atlassian.com/x/BYfsNg\u0022 target=\u0022_blank\u0022\u003eLearn more\u003c/a\u003e",d);e.closeMessages(["utf8-validation-failed"]);e.handleMessage("utf8-validation-failed",{title:"Unsupported character",type:"error",message:d})}else{d=
"Failed to save the comment. Please try again later.";if(c)g.showError(g.parseError(c,d),g.Location.FLAG);else{e.closeMessages(["server-offline"]);e.handleMessage("server-offline",{type:"error",message:d})}}q.logError("Error saving comment",a)},cancelComment:function(){b.Rte.Content.editorResetContentChanged();return require("confluence-quick-edit/quick-edit").deactivateEditor()},proceedWithActivation:function(){var a=new f.Deferred,c=b.Rte&&b.Rte.getEditor();if(c){if(c.isDirty()&&!i.Editor.isEmpty()&&
!confirm("Your comment will be lost."))return a.reject();a=j.cancelComment()}else a.resolve();return a}};return j});require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/handlers/comment","AJS.Confluence.QuickEdit.QuickComment");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-comment-initial', location = 'jscripts/handlers/comment/top-level.js' */
define("confluence-quick-edit/handlers/comment/top-level","jquery confluence/dark-features confluence-editor-loader/editor-loader confluence/api/event confluence/api/logger confluence-editor/profiles confluence-quick-edit/handlers/comment confluence-quick-edit/quick-edit confluence-quick-edit/handlers/shortcut window".split(" "),function(a,i,j,k,e,l,b,f,m,g){function n(a){b.preInitialise(a)}function o(c){a("#comments-section").one("click",".quick-comment-prompt",d.activateEventHandler);a("#rte-savebar").scrollWindowToElement();
b.postInitialise(c)}function p(){if(!j.resourcesLoaded()){k.trigger("analytics",{name:"rte.quick-edit.top-comment.spinner"});a(".quick-comment-prompt").hide();a(".quick-comment-loading-container").fadeIn().spin("medium")}}function q(b){b.preventDefault();g.location=a("#add-comment-rte").attr("href")}var r=i.isEnabled("editor.slow.comment.disable"),d={commentShortcut:m.createShortcut("m",".quick-comment-prompt"),activateEventHandler:function(c){c.preventDefault();b.proceedWithActivation().done(function(){var c=
b.createSaveHandler(b.delegatingSaveCommentHandler,b.saveCommentErrorHandler),h=a("form[name=inlinecommentform]");return f.activateEditor({preActivate:p,$container:h.closest(".quick-comment-container"),$form:h,preInitialise:n,saveHandler:c,cancelHandler:b.cancelHandler,postInitialise:o,plugins:l.createProfileForCommentEditor().plugins,additionalResources:["wrc!comment-editor"],timeoutResources:b.timeout}).fail(function(c){e.logError("activateEventHandler failed because of: "+c);if(c==="READ_ONLY"||
r){b.showLoadingEditorErrorMessage(c);a("#comments-section").one("click",".quick-comment-prompt",d.activateEventHandler)}else g.location=a("#add-comment-rte").attr("href")})}).fail(function(){a("#comments-section").one("click",".quick-comment-prompt",d.activateEventHandler)})},enable:function(){a("#comments-section").one("click",".quick-comment-prompt",d.activateEventHandler);a(".quick-comment-prompt").on("keydown",function(b){b.keyCode===a.ui.keyCode.ENTER&&d.activateEventHandler(b)});a("#add-comment-rte").removeClass("full-load");
this.commentShortcut.bind()},disable:function(){a("#comments-section").off("click",".quick-comment-prompt");this.commentShortcut.unbind()}};f.register(d);return{bindCommentAreaFallbackHandler:function(){a("#comments-section").delegate(".quick-comment-prompt","click",q)},cancelComment:function(){e.log("'AJS.Confluence.QuickEdit.QuickComment.TopLevel.cancelComment' is deprecated in 5.7, consider using 'AJS.Confluence.QuickEdit.QuickComment.cancelComment' instead.");return b.cancelComment()}}});
require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/handlers/comment/top-level","AJS.Confluence.QuickEdit.QuickComment.TopLevel");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-comment-initial', location = 'jscripts/handlers/comment/reply.js' */
define("confluence-quick-edit/handlers/comment/reply","confluence/root confluence/templates wrm/context-path confluence/dark-features confluence-editor-loader/editor-loader confluence/api/event confluence/api/logger confluence/meta confluence-editor/profiles confluence-quick-edit/handlers/comment confluence-quick-edit/quick-edit jquery".split(" "),function(l,m,n,o,p,g,h,i,q,b,j,a){function r(a){b.preInitialise(a);a.$container.scrollWindowToElement()}function s(a){b.postInitialise(a)}function k(a){var b=
a.attr("id").match(/comment-(\d+)/),e=0;if(b)e=parseInt(b[1],10);else{g.trigger("analytics",{name:"rte.quick-edit.get-reply-parent.failed"});h.logError("replyHandler: activateEventHandler - could not extract a parent comment Id from the comment id "+a.attr("id"))}return e}function t(){a(".comment.reply").closest(".comment-threads").remove()}var d,u=o.isEnabled("editor.slow.comment.disable");d={activateEventHandler:function(d){var f=this;d.preventDefault();d.stopPropagation();if(a(f).attr("reply-disabled"))return false;
b.proceedWithActivation().done(function(){var e=a(f).closest("div.comment"),c;c=a(".quick-comment-container img.userLogo");c=b.createCommenterParam(c);c={contentId:l.getContentId(),parentCommentId:k(e),commenter:c,context:{contextPath:n(),staticResourceUrlPrefix:i.get("static-resource-url-prefix")}};c=a(m.Comments.displayReplyEditorLoadingContainer(c));var d=a(".quick-comment-loading-container",c);d.hide();e.after(c);if(p.resourcesLoaded())g.trigger("analytics",{name:"rte.quick-edit.reply-comment.no-spinner"});
else{g.trigger("analytics",{name:"rte.quick-edit.reply-comment.spinner"});e.after(c);d.fadeIn();d.spin("medium");d[0].scrollIntoView()}c=a(f).closest(".comment-thread").find(".quick-comment-container");i.set("form-name",a("form",c).attr("name"));j.activateEditor({$container:c,$form:a("form.quick-comment-form[name=threadedcommentform]"),preInitialise:r,postInitialise:s,saveHandler:b.createSaveHandler(b.delegatingSaveCommentHandler,b.saveCommentErrorHandler),cancelHandler:b.cancelHandler,plugins:q.createProfileForCommentEditor().plugins,
postDeactivate:t,additionalResources:["wrc!comment-editor"],timeoutResources:b.timeout}).fail(function(){u?b.showLoadingEditorErrorMessage():window.location=a("#reply-comment-"+k(e)).attr("href")});a(f).attr("reply-disabled",true)})},enable:function(){a("#comments-section").delegate(".action-reply-comment","click",d.activateEventHandler)},disable:function(){a("#comments-section").undelegate(".action-reply-comment","click")}};j.register(d);return{cancelComment:function(){h.log("'AJS.Confluence.QuickEdit.QuickComment.Reply.cancelComment' is deprecated in 5.7, consider using 'AJS.Confluence.QuickEdit.QuickComment.cancelComment' instead.");
return b.cancelComment()}}});require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/handlers/comment/reply","AJS.Confluence.QuickEdit.QuickComment.Reply");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-comment-initial', location = 'jscripts/handlers/comment/edit.js' */
define("confluence-quick-edit/handlers/comment/edit","confluence/root confluence/templates wrm/context-path confluence/dark-features confluence/api/logger confluence/meta confluence-editor/profiles confluence-quick-edit/handlers/comment confluence-quick-edit/quick-edit jquery".split(" "),function(j,k,l,m,n,g,o,b,i,d){function p(a){b.preInitialise(a);a.$container.scrollWindowToElement()}function q(a){b.postInitialise(a)}function h(a){return(a=a.attr("id").match(/comment-(\d+)/))?parseInt(a[1],10):
0}function r(a){var b=new d.Deferred;d.ajax({url:l()+"/rest/api/content/"+a+"?expand=body.editor",cache:false}).done(function(a){!a||!a.body||!a.body.editor?b.reject("invalid response from loading comment rest endpoint"):b.resolve({editorContent:a.body.editor.value})}).fail(function(){b.reject("error fetching content")});return b}function s(){var a=d(".comment.edit");a.prev(".comment").show();a.remove()}var t=m.isEnabled("editor.slow.comment.disable"),f={activateEventHandler:function(a){var f=this;
a.preventDefault();a.stopPropagation();b.proceedWithActivation().done(function(){var a=d(f).closest("div.comment"),c;c=a.find(".author .confluence-userlink");var e=a.find(".comment-user-logo img.userLogo");c=b.createCommenterParam(e,c.attr("data-username"),c.text());c={contentId:j.getContentId(),commentId:h(a),commenter:c,context:{contextPath:g.get("context-path"),staticResourceUrlPrefix:g.get("static-resource-url-prefix")},mode:"edit"};c=d(k.Comments.displayEditEditorContainer(c));e=d(".quick-comment-loading-container",
c);a.hide();a.after(c);e.fadeIn().spin("medium");e[0].scrollIntoView();c=a.next(".quick-comment-container");g.set("form-name",d("form",c).attr("name"));i.activateEditor({$container:c,$form:d("form.quick-comment-form[name=editcommentform]"),fetchContent:r(h(a)),preInitialise:p,postInitialise:q,saveHandler:b.createSaveHandler(b.delegatingSaveCommentHandler,b.saveCommentErrorHandler),cancelHandler:b.cancelHandler,plugins:o.createProfileForCommentEditor().plugins,postDeactivate:s,additionalResources:["wrc!comment-editor"],
timeoutResources:b.timeout}).fail(function(){t?b.showLoadingEditorErrorMessage():window.location=d("#edit-comment-"+h(a)).attr("href")})})},enable:function(){d("#comments-section").delegate(".comment-action-edit","click",f.activateEventHandler)},disable:function(){d("#comments-section").undelegate(".comment-action-edit","click")}};i.register(f);return{cancelComment:function(){n.log("'AJS.Confluence.QuickEdit.QuickComment.Edit.cancelComment' is deprecated in 5.7, consider using 'AJS.Confluence.QuickEdit.QuickComment.cancelComment' instead.");
return b.cancelComment()}}});require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/handlers/comment/edit","AJS.Confluence.QuickEdit.QuickComment.Edit");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-comment-initial', location = 'jscripts/editor-comment-manager.js' */
define("confluence-quick-edit/editor-comment-manager",["ajs","jquery","confluence/legacy"],function(d,e,j){return function(){function k(a,f,l,i,b,g,h){d.trigger("analytics",{name:"confluence.page.comment.create",data:{pageID:d.Meta.get("page-id")}});a={type:"POST",url:a,contentType:"application/x-www-form-urlencoded; charset=UTF-8",data:{html:f,watch:l,uuid:i},dataType:"json",cache:!1,headers:{"X-Atlassian-Token":"no-check"},success:function(a){g(a)},error:function(a,c,b){c=c+": "+b;a.responseText&&
(c=c+" - "+a.responseText);h(c,a)},timeout:12E4};b&&b.id&&(a.headers["X-Atlassian-Captcha-Id"]=b.id,a.headers["X-Atlassian-Captcha-Response"]=b.response);e.ajax(a)}return{addComment:function(a,f,d,i,b,g,h,e,c){j.Editor.CommentManager.saveComment(a,f,d,function(a){j.CommentDisplayManager.addComment(h,a,g);e(a)},c)},saveComment:function(a,f,e,i,b,g,h,j){var c=null,c=f?d.contextPath()+"/rest/tinymce/1/content/"+a+"/comments/"+f+"/comment?actions=true":d.contextPath()+"/rest/tinymce/1/content/"+a+"/comment?actions=true";
k(c,e,i,b,g,h,j)},updateComment:function(a,f,e,i,b,g,h){a=d.contextPath()+"/rest/tinymce/1/content/"+a+"/comments/"+f+"?actions=true";k(a,e,i,null,b,g,h)}}}});require("confluence/module-exporter").safeRequire("confluence-quick-edit/editor-comment-manager",function(d){var e=require("confluence/legacy");require("ajs").bind("init.rte",function(){e.Editor.CommentManager=d()})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:editor-view-resources', location = 'jscripts/view-content/pushed-navigation-util.js' */
define("confluence-quick-edit/view-content/pushed-navigation-util",["ajs","jquery","window"],function(d,f,a){function c(){return d.Rte&&d.Rte.getEditor()&&!!f("#editpageform").length}var b=a.location.hash,e=a.location.search;return{isInEditPage:c,filterPreviewHashChange:function(){var g=c()||a.history.pushState||!(a.location.hash&&0===a.location.hash.indexOf("#!"))&&!(b&&0===b.indexOf("#!"));b=a.location.hash;return g},filterPreviewNavigationEvent:function(){var b=c()||!/[?&]preview=([^&]*)/.test(a.location.search)&&
!/[?&]preview=([^&]*)/.test(e);e=a.location.search;return b}}});require("confluence/module-exporter").exportModuleAsGlobal("confluence-quick-edit/view-content/pushed-navigation-util","Confluence.Editor.PushedNavUtil");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:editor-view-resources', location = 'jscripts/view-content/pushed-navigation.js' */
define("confluence-quick-edit/view-content/pushed-navigation","jquery underscore window confluence/legacy confluence/api/event confluence-quick-edit/view-content/pushed-navigation-util".split(" "),function(b,n,a,f,g,c){function d(){if(h){if(e.split("#")[0]!=a.location.href.split("#")[0]){b(a).unbind("popstate",k);a.location.reload()}}else e.split("#")[0]==a.location.href.split("#")[0]&&e.split("#")[1]!=="editor"||a.location.reload()}function i(){e=a.location.href}function l(){var b=f.Editor.Drafts.unloadMessage();
if(b){f.Editor.Drafts.unBindUnloadMessage();if(confirm(b+"\n\n"+"Press OK to continue, or Cancel to stay on the current page.")){g.trigger("analytics",{name:"rte.quick-edit.confirmation.leaving"});d()}else{g.trigger("analytics",{name:"rte.quick-edit.confirmation.staying"});if(h){j=true;a.history.forward()}else a.location.hash="editor";f.Editor.Drafts.bindUnloadMessage()}}else d()}function o(){c.isInEditPage()?a.location.hash!=="#editor"&&l():d()}function k(){j?j=false:c.isInEditPage()?l():d()}function m(a,b){return function(){n.every(b,
function(a){return a()})&&a()}}var h=!!a.history.pushState,j=false,e=a.location.href,p=[c.filterPreviewHashChange],q=[c.filterPreviewNavigationEvent];g.bind("rte-quick-edit-enable-handlers",function(){a.location.hash==="#editor"&&b("#editPageLink").click()});return function(){i();if(h){b(a).bind("popstate",m(k,q));b(a).bind("rte-quick-edit-push-state",i)}else{b(a).bind("hashchange",m(o,p));b(a).bind("rte-quick-edit-push-hash",i)}}});
require("confluence/module-exporter").safeRequire("confluence-quick-edit/view-content/pushed-navigation",function(b){require("ajs").toInit(function(){setTimeout(b,0)})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.splitchunk.5b7fdbd666', location = 'aui.chunk.d94be0a9210a700c896f--755920dedd8cdb7aa16c.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.splitchunk.5b7fdbd666"],{fVKi:function(t,n,r){"use strict";r.r(n);var i=r("JFi+");n.default=function(t){function n(n){return e((function(){return t.hasAttribute(n)}),n+" wasn't defined")}function r(r){if(!n(r))return!1;var i=t.getAttribute(r);return e((function(){return document.getElementById(i)}),'an element with id set to "'+i+'" was not found')}function e(n,r){return!!n()||(t?i.error(r,t):i.error(r),!1)}return{attributeExists:n,refersTo:r,satisfiesRules:e,ariaControls:function(){return r("aria-controls")},ariaOwns:function(){return r("aria-owns")}}}}}]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.splitchunk.8a641c03a4', location = 'aui.chunk.3242b57cec7cf8249eaf--cda411b08634c5067d51.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.splitchunk.8a641c03a4"],{kdZR:function(t,e,u){"use strict";function n(t){return null!==t}function i(t,e,u){u?t.setAttribute(e,""):t.removeAttribute(e)}function l(t,e){const u=t=>t.toLowerCase()===e.toLowerCase(),n=null===e,i=!n&&!t.values.filter(u).length;return n?t.hasOwnProperty("missingDefault")?t.missingDefault:null:i?t.hasOwnProperty("invalidDefault")?t.invalidDefault:t.hasOwnProperty("missingDefault")?t.missingDefault:null:t.values.length?t.values.filter(u)[0]:null}function o(t,e,u){t.setAttribute(e.attribute,u)}u.r(e),u.d(e,"computeBooleanValue",(function(){return n})),u.d(e,"setBooleanAttribute",(function(){return i})),u.d(e,"computeEnumValue",(function(){return l})),u.d(e,"setEnumAttribute",(function(){return o})),e.default={computeBooleanValue:n,setBooleanAttribute:i,computeEnumValue:l,setEnumAttribute:o}}}]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.splitchunk.d727dd207f', location = 'aui.chunk.de01c5860af45d043e5b--b445726f8c43e9b4260b.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.splitchunk.d727dd207f"],{VtiI:function(i,n,s){"use strict";s.r(n);s("FStl"),s("Q0fs"),s("nqD9"),s("NjKm")}}]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.splitchunk.543254b237', location = 'aui.chunk.2496bb5d5821eb9f97e3--e3bee42df72c5f62f136.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.splitchunk.543254b237"],{P3bb:function(e,t,n){"use strict";n.r(t);var i=n("+x/D"),o=n("vbuG"),r=n("TmQU"),u=n("kdZR"),s=n("fVKi"),a=n("KloK"),c=n("jEnx"),l=n("4dFR"),d=n("TK3y"),f=n("/soZ"),b=n("I7um"),p=n("ih6f");function g(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);t&&(i=i.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,i)}return n}function m(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?g(Object(n),!0).forEach((function(t){O(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):g(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function O(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function j(e,t){Object(f.doIfTrigger)(e,(function(n){n.setAttribute("aria-expanded","false"),t.setAttribute("aria-expanded",e.open)})),Object(f.setTrigger)(e,t)}function v(e,t){if(e._auiAlignment)e._auiAlignment.changeTarget(t),e._auiAlignment.enable();else{let n={overflowContainer:"viewport"===e.getAttribute("contained-by")?"viewport":"window",positionFixed:!1,eventsEnabled:!0};n=m(m({},n),{onCreate:()=>{L(e)&&Object(p.default)().enter(Object(i.default)(e),Object(i.default)(t))},onEvents:{enabled:()=>{L(e)&&Object(p.default)().enter(Object(i.default)(e))},disabled:()=>{L(e)&&Object(p.default)().exit(Object(i.default)(e))}}}),e._auiAlignment=new o.default(e,t,n)}}function h(e){e._auiAlignment&&(e._auiAlignment.destroy(),delete e._auiAlignment)}function T(e){return Object(d.default)(e).get("mouse-inside")||e.contains(document.activeElement)}function _(e,t){var n=t.currentTarget;n&&(j(e,n),v(e,n)),e.open||(e.open=!0),clearTimeout(e._closingTimeout)}function A(){let e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0;return function(t){t.open&&!Object(c.default)(t).isPersistent()&&(clearTimeout(t._closingTimeout),t._closingTimeout=setTimeout((function(){T(t)||(t.open=!1),t._closingTimeout=null}),e))}}const y={click(e,t){e.open&&!Object(c.default)(e).isPersistent()?e.open=!1:(j(e,t.currentTarget),e.open=!0),clearTimeout(e._closingTimeout)},mouseenter:_,mouseleave:A(1e3),focus:_,blur:A(0)};function E(e){var t=e.currentTarget;Object(d.default)(t).set("mouse-inside",!0),t.message({type:"mouseenter"})}function w(e){var t=e.currentTarget;Object(d.default)(t).set("mouse-inside",!1),t.message({type:"mouseleave"})}function P(e){var t=e.currentTarget;"hover"===t.respondsTo&&A(1e3)(t)}function x(e){Object(d.default)(e).set("mouse-inside",void 0),e.removeEventListener("mouseenter",E),e.removeEventListener("mouseleave",w),e.removeEventListener("blur",P),"hover"===e.respondsTo&&(Object(d.default)(e).set("mouse-inside",!1),e.addEventListener("mouseenter",E),e.addEventListener("mouseleave",w),e.addEventListener("blur",P))}function V(e,t){return"".concat(e,".nested-layer-").concat(t)}function D(e){return Object(c.default)(e).show(),!0===Object(c.default)(e).isVisible()}function I(e){return Object(c.default)(e).hide(),!1===Object(c.default)(e).isVisible()}function L(e){return"hover"!==e.respondsTo}function k(e){e.__initialised||(Object(c.default)(e),Object(i.default)(e).on({["".concat(c.EVENT_PREFIX,"show")]:function(){const e=this;!function(e){let t=Object(i.default)(e);const n=e.id,o=e=>t.find(Object(f.getTrigger)(e.target)).length<1;Object(i.default)(document).on(V("aui-layer-show",n),(e=>{o(e)||t.attr("persistent","")})).on(V("aui-layer-hide",n),(e=>{o(e)||t.removeAttr("persistent")})).on(V("select2-opening",n),(()=>{t.attr("persistent","")})).on(V("select2-close",n),(()=>{setTimeout((()=>{t.removeAttr("persistent")}),150)}))}(e),Object(f.doIfTrigger)(e,(function(t){v(e,t),t.setAttribute("aria-expanded","true")}))},["".concat(c.EVENT_PREFIX,"hide")]:function(){const e=this;!function(e){const t=e.id;Object(i.default)(document).off(V("aui-layer-hide",t)).off(V("aui-layer-show",t)).off(V("select2-opening",t)).off(V("select2-close",t))}(e.id),e.ownerDocument.body.contains(e)?function(e){e._auiAlignment&&e._auiAlignment.disable()}(e):h(e),Object(f.doIfTrigger)(e,(function(e){e.setAttribute("aria-expanded","false")})),Object(f.setTrigger)(e,null)}}),e.__initialised=!0)}const R={attribute:"responds-to",values:["toggle","hover"],missingDefault:"toggle",invalidDefault:"toggle"},F=Object(l.default)("aui-inline-dialog",{prototype:{get open(){return Object(c.default)(this).isVisible()},set open(e){this.__propUpdate=!0,e?D(this):I(this)},get persistent(){return this.hasAttribute("persistent")},set persistent(e){u.default.setBooleanAttribute(this,"persistent",e)},get respondsTo(){var e=R.attribute;return u.default.computeEnumValue(R,this.getAttribute(e))},set respondsTo(e){const t=this.respondsTo;u.default.setEnumAttribute(this,R,e),t!==this.respondsTo&&x(this)},message:function(e){return function(e,t){var n={toggle:["click"],hover:["mouseenter","mouseleave","focus","blur"]}[e.respondsTo];n&&n.indexOf(t.type)>-1&&y[t.type](e,t)}(this,e),this}},attributes:{open:function(e,t){if(k(e),e.__propUpdate)delete e.__propUpdate;else{if("created"===t.type){D(e)||Object(u.setBooleanAttribute)(e,"open",!1)}if("removed"===t.type){I(e)||Object(u.setBooleanAttribute)(e,"open",!0)}}},"responds-to":function(e,t){u.default.computeEnumValue(R,t.oldValue)!==u.default.computeEnumValue(R,t.newValue)&&x(e)}},created:k,attached:function(e){Object(s.default)(e).attributeExists("id"),e.setAttribute("tabindex",0),e.setAttribute("role","dialog"),Object(f.doIfTrigger)(e,(function(t){t.setAttribute("aria-expanded",e.open)})),Object(f.forEachTrigger)(e,(function(e){e.setAttribute("aria-haspopup","true")})),x(e)},detached:function(e){Object(b.ifGone)(e).then((()=>{h(e),Object(f.forEachTrigger)(e,(function(e){e.removeAttribute("aria-haspopup"),e.removeAttribute("aria-expanded")}))}))},template:function(e){Object(i.default)('<div class="aui-inline-dialog-contents"></div>').append(e.childNodes).appendTo(e)}});Object(r.default)("aui/inline-dialog2",F),Object(a.default)("InlineDialog2",F),t.default=F},Rvtc:function(e,t,n){"use strict";n.r(t);n("AehQ"),n("THZ5"),n("VtiI");var i=n("P3bb");n.d(t,"InlineDialogEl",(function(){return i.default}))}}]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.component.inline-dialog2', location = 'aui.chunk.32425e0082182a521159--1c7652573c89388165f5.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.component.inline-dialog2"],[],[["Rvtc","runtime","aui.splitchunk.vendors--894c8113d9","aui.splitchunk.vendors--084821f40b","aui.splitchunk.0d131bcbf1","aui.splitchunk.fbbef27525","aui.splitchunk.444efc83be","aui.splitchunk.739b9ec8cc","aui.splitchunk.056561461c","aui.splitchunk.949297951c","aui.splitchunk.dd803a46b4","aui.splitchunk.994e478d48","aui.splitchunk.e54c7c7304","aui.splitchunk.fb15cffa72","aui.splitchunk.f1e06f97a4","aui.splitchunk.f673ef53ac","aui.splitchunk.8659b532c1","aui.splitchunk.d0110a864f","aui.splitchunk.afa5039e04","aui.splitchunk.bff3715233","aui.splitchunk.c750721820","aui.splitchunk.6d6f245ed3","aui.splitchunk.084821f40b","aui.splitchunk.5b8c290363","aui.splitchunk.baa83dbaf9","aui.splitchunk.36cd9d521c","aui.splitchunk.d925afe2c0","aui.splitchunk.5b7fdbd666","aui.splitchunk.8a641c03a4","aui.splitchunk.d727dd207f","aui.splitchunk.543254b237"]]]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-previews:confluence-previews-jquery', location = '/js/confluence/preview-support.js' */
define("cp/confluence/preview-support",["ajs"],function(a){var d=function(){var e=b.SELECTOR_STRINGS.IMAGE+", "+b.SELECTOR_STRINGS.LINK_IMAGE+", "+b.SELECTOR_STRINGS.EXTERNAL_IMAGE;if(b.isPDFSupported()){e+=", "+b.SELECTOR_STRINGS.PDF+", "+b.SELECTOR_STRINGS.LINK_PDF}return e};var c=function(){return b.SELECTOR_STRINGS.IMAGE+", "+b.SELECTOR_STRINGS.EXTERNAL_IMAGE+", "+b.SELECTOR_STRINGS.FILE+", "+b.SELECTOR_STRINGS.LINK_FILE+", "+b.SELECTOR_STRINGS.ATTACHMENT_MACRO};var b={SELECTOR_STRINGS:{IMAGE:"img.confluence-embedded-image[data-linked-resource-id]",EXTERNAL_IMAGE:"img.confluence-embedded-image.confluence-external-resource",PDF:"a.confluence-embedded-file[data-nice-type='PDF Document']",LINK_IMAGE:"a[data-linked-resource-type='attachment'][data-nice-type='Image']",LINK_PDF:"a[data-linked-resource-type='attachment'][data-nice-type='PDF Document']",FILE:"a.confluence-embedded-file",LINK_FILE:"a[data-linked-resource-type='attachment']",FILE_OVERLAY:"span.confluence-embedded-file-wrapper .overlay",ATTACHMENT_MACRO:".plugin_attachments_container a.previewAttachmentLink"},VIEW_MODE:{FULL:"full",COMMENT:"comment",SIMPLE:"simple"},isPDFSupported:function(){return a.DarkFeatures.isEnabled("pdf-preview")},getFileSelectorString:function(){if(a.DarkFeatures.isEnabled("previews.trigger-all-file-types")){return c()}else{return d()}}};return b});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-previews:confluence-previews-jquery', location = '/js/confluence/preview.js' */
define("cp/confluence/preview",["underscore","jquery","cp/confluence/preview-support","ajs","confluence/jsUri"],function(k,e,f,i,b){var j,l;e(document).ready(g);function g(){var o=e(f.getFileSelectorString());o.off("click.fb");e(document.body).off("click.filePreviews");e(document.body).on("click.filePreviews",f.getFileSelectorString(),n);e(document.body).on("click.filePreviews",f.SELECTOR_STRINGS.FILE_OVERLAY,h);e(window).on("popstate",function(){if(!l&&i.DarkFeatures.isEnabled("previews.sharing.pushstate")){a()}});a()}function n(o){if(!o.altKey&&!o.ctrlKey&&!o.metaKey&&!e(this).parent().closest("a, #draft-changes-dialog, #cp-annotations, .diff-block-context, .diff-block-target").length){o.preventDefault();m(this)}}function h(p){p.preventDefault();var o=e(p.target);var q=o.closest("span.confluence-embedded-file-wrapper");m(q.find(f.SELECTOR_STRINGS.IMAGE+","+f.SELECTOR_STRINGS.FILE),undefined,undefined,o.closest(".comment-count-overlay").length>0)}function m(q,r,o,x){if(!l){var s="confluence-previews-css";var p="media-viewer";var y="viewattachments";WRM.require("wr!com.atlassian.confluence.plugins.confluence-previews:"+s);var z=e.Deferred(),u=z.promise();var t=setInterval(function(){for(var D=0;D<document.styleSheets.length;D++){var E=document.styleSheets[D];if(E.href&&(E.href.indexOf(s)!==-1||E.href.indexOf(p)!==-1||E.href.indexOf(y)!==-1)){w();return}if(document.styleSheets[D].imports&&document.all&&!window.atob){for(var C=0;C<document.styleSheets[D].imports.length;C++){if(document.styleSheets[D].imports[C].href.indexOf(s)!==-1){w();return}}}}},100);var w=function(){z.resolve();clearInterval(t)};var B=i.Meta.get("static-resource-url-prefix");var A="com.atlassian.confluence.plugins.confluence-previews:mediaviewer-chunks";var v=B+"/download/resources/"+A+"/";l=e.when(WRM.require(["wr!com.atlassian.confluence.plugins.confluence-previews:confluence-previews-js","wrc!media-viewer"]),u);l.done(function(){j=require("cp/confluence/file-preview-loader");d(q,r,o,x)});Confluence.PageLoadingIndicator(e("body")).showUntilResolved(l,"We were unable to load the file previewer")}else{l.done(k.partial(d,q,r,o,x))}return l}var d=function(q,p,s,t){var o=e("#content");var r=e(q).closest(".comment,.cq-content,.ic-content");if(!p){if(!(o.hasClass("page")||o.hasClass("blogpost"))){p=f.VIEW_MODE.SIMPLE}else{if(r.length){p=f.VIEW_MODE.COMMENT}}}if(p===f.VIEW_MODE.COMMENT){j.showPreviewerForComment(q,r,p,t)}else{if(p===f.VIEW_MODE.SIMPLE){j.showPreviewerForSingleFile(q,p,s)}else{p=f.VIEW_MODE.FULL;j.showPreviewer(undefined,q,p,t)}}};function a(){if(c()){var q=new b(window.location.href);var p=window.history.pushState&&i.DarkFeatures.isEnabled("previews.sharing.pushstate");if(q.getQueryParamValue("preview")&&!p){var r="#!/preview"+q.getQueryParamValue("preview");var o=decodeURIComponent(q.deleteQueryParam("preview").setAnchor(r).toString());if(window.history.replaceState){window.history.replaceState({},"",o)}else{window.location=o}}else{if(q.anchor().indexOf("!/preview")===0&&p){var o;if(q.getQueryParamValue("preview")){o=q.setAnchor("")}else{o=q.addQueryParam("preview",q.anchor().substr("!/preview".length,q.anchor().length)).setAnchor("")}window.history.replaceState({},"",o)}}m()}}function c(){var o=new b(window.location.href);return i.DarkFeatures.isEnabled("previews.sharing")&&(o.getQueryParamValue("preview")||(o.anchor()&&o.anchor().indexOf("!/preview")===0))}return{loadConfluencePreviews:m}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-previews:confluence-previews-jquery', location = '/js/confluence/jquery-previewer.js' */
require(["cp/confluence/preview","jquery"],function(b,a){a.fn.previewer=function(c){if(!a(this).length){return this}var d=a.extend({},c);return this.each(function(){var f=a(this);var e=f.closest("li");var g=d.src||f.attr("data-image-src")||f.attr("src");if(g){f.click(function(j){var i={src:g,type:d.type,thumbnail:g,title:d.title||e.attr("data-file-name")||g,id:e.attr("data-attachment-id"),ownerId:e.attr("data-owner-id")};var h=b.loadConfluencePreviews([i],d.viewMode||"simple",d.from||"custom");d.zindex&&h.done(function(){a(".cp-container").css({"z-index":d.zindex})})})}})}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'org.randombits.confluence.toc:toc-plugin-analytics', location = 'net/customware/confluence/plugin/toc/analytics/analytics.js' */
AJS.toInit(function(){AJS.$(".toc-macro a").click(function(){AJS.trigger("analyticsEvent",{name:"confluence.toc-macro.heading-click"})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'org.randombits.confluence.toc:client-side-toc-resources', location = 'net/customware/confluence/plugin/toc/js/client-side-toc.js' */
require(["ajs"],function(p){p.toInit(function(d){function q(b){var a=d({});a.data("precedenceLevel",b);return a}function k(b){return d(b).data("precedenceLevel")}function m(b,a,c){if(0===b.length)return d();var e=b.map(k).reduce(function(f,l){return Math.min(f,l)});e!==k(b[0])&&b.unshift(q(e));var g=a.createTocLevelContainer(),h={subElements:[],currentItem:void 0,outline:void 0,flush:function(){0<this.subElements.length&&this.currentItem&&(m(this.subElements,a,this.outline).appendTo(this.currentItem),
this.subElements=[])},add:function(f){this.subElements.push(f)},resetItem:function(f){this.outline=(c||[]).slice(0);this.outline.push(f);this.currentItem=a.createTocItemContainer();this.currentItem.appendTo(g);return this.currentItem}},n=0;b.forEach(function(f){if(e===k(f))if(n++,h.flush(),h.resetItem(n),f.textContent){var l=h.outline.join(".");d(Confluence.Plugins.TableOfContents.Client.tocItemBody({outline:l,linkHref:"#"+f.id,linkText:f.textContent})).appendTo(h.currentItem)}else h.currentItem.addClass("toc-empty-item");
else h.add(f)});h.flush();0===c.length&&a.decorateToc&&a.decorateToc(g);return g}function r(b){return{createTocLevelContainer:function(){return this.createTocItemContainer()},createTocItemContainer:function(){return d(Confluence.Plugins.TableOfContents.Client.flatStyleTocItemContainer({cssClass:"toc-item-container"}))},decorateToc:function(a){function c(e,g){(e=e in b?b[e]:g)&&d(Confluence.Plugins.TableOfContents.Client.flatStyleTocSeparator({separator:e})).appendTo(a)}c("preseparator","[ ");d(a).find("span.toc-item-body").each(function(e,
g){0<e&&c("midseparator"," ] [ ");d(g).appendTo(a)});c("postseparator"," ]");d(a).find(".toc-item-container").remove()}}}function t(b){return{createTocLevelContainer:function(){return d(Confluence.Plugins.TableOfContents.Client.listStyleTocLevelContainer({cssliststyle:b.cssliststyle,csslistindent:b.csslistindent}))},createTocItemContainer:function(){return d(Confluence.Plugins.TableOfContents.Client.listStyleTocItemContainer())}}}function u(b){var a;b.includeheaderregex&&(a=new RegExp(b.includeheaderregex));
var c;b.excludeheaderregex&&(c=new RegExp(b.excludeheaderregex));return function(){var e=d(this).text();return a&&!a.test(e)||c&&c.test(e)?!1:!0}}function v(b,a){var c=a.headerelements,e=c.split(",");a=u(a);c=d("#content .wiki-content").find(c).filter(a).each(function(){var g=e.indexOf(this.nodeName);d(this).data("precedenceLevel",g)}).toArray();return m(c,b,[])}d(".client-side-toc-macro").each(function(){var b=d(this),a=b.data()||{};var c="flat"===a.structure?r(a):t(a);!0!==a.numberedoutline&&b.addClass("hidden-outline");
b.html(v(c,a))})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'org.randombits.confluence.toc:client-side-toc-resources', location = 'net/customware/confluence/plugin/toc/templates/client.soy' */
// This file was automatically generated from client.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Plugins.TableOfContents.Client.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Plugins == 'undefined') { Confluence.Plugins = {}; }
if (typeof Confluence.Plugins.TableOfContents == 'undefined') { Confluence.Plugins.TableOfContents = {}; }
if (typeof Confluence.Plugins.TableOfContents.Client == 'undefined') { Confluence.Plugins.TableOfContents.Client = {}; }


Confluence.Plugins.TableOfContents.Client.tocItemBody = function(opt_data, opt_ignored) {
  return '<span class="toc-item-body" data-outline="' + soy.$$escapeHtml(opt_data.outline) + '"><span class="toc-outline">' + soy.$$escapeHtml(opt_data.outline) + '</span><a href="' + soy.$$escapeHtml(opt_data.linkHref) + '" class="toc-link">' + soy.$$escapeHtml(opt_data.linkText) + '</a></span>';
};
if (goog.DEBUG) {
  Confluence.Plugins.TableOfContents.Client.tocItemBody.soyTemplateName = 'Confluence.Plugins.TableOfContents.Client.tocItemBody';
}


Confluence.Plugins.TableOfContents.Client.listStyleTocLevelContainer = function(opt_data, opt_ignored) {
  return '<ul style="' + ((opt_data.cssliststyle) ? ' list-style: ' + soy.$$escapeHtml(opt_data.cssliststyle) + ';' : '') + ((opt_data.csslistindent) ? ' padding-left: ' + soy.$$escapeHtml(opt_data.csslistindent) + ';' : '') + '"></ul>';
};
if (goog.DEBUG) {
  Confluence.Plugins.TableOfContents.Client.listStyleTocLevelContainer.soyTemplateName = 'Confluence.Plugins.TableOfContents.Client.listStyleTocLevelContainer';
}


Confluence.Plugins.TableOfContents.Client.listStyleTocItemContainer = function(opt_data, opt_ignored) {
  return '<li></li>';
};
if (goog.DEBUG) {
  Confluence.Plugins.TableOfContents.Client.listStyleTocItemContainer.soyTemplateName = 'Confluence.Plugins.TableOfContents.Client.listStyleTocItemContainer';
}


Confluence.Plugins.TableOfContents.Client.flatStyleTocItemContainer = function(opt_data, opt_ignored) {
  return '<span class="' + soy.$$escapeHtml(opt_data.cssClass) + '"></span>';
};
if (goog.DEBUG) {
  Confluence.Plugins.TableOfContents.Client.flatStyleTocItemContainer.soyTemplateName = 'Confluence.Plugins.TableOfContents.Client.flatStyleTocItemContainer';
}


Confluence.Plugins.TableOfContents.Client.flatStyleTocSeparator = function(opt_data, opt_ignored) {
  return '<span class="toc-separator">' + soy.$$escapeHtml(opt_data.separator) + '</span>';
};
if (goog.DEBUG) {
  Confluence.Plugins.TableOfContents.Client.flatStyleTocSeparator.soyTemplateName = 'Confluence.Plugins.TableOfContents.Client.flatStyleTocSeparator';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.analytics.analytics-client:js-events', location = 'js/store-1.3.1.js' */
(function(){var l={},h=window,k=h.document,c="localStorage",n="globalStorage",d="__storejs__",g;l.disabled=false;l.set=function(e,o){};l.get=function(e){};l.remove=function(e){};l.clear=function(){};l.transact=function(e,o){var p=l.get(e);if(typeof p=="undefined"){p={}}o(p);l.set(e,p)};l.serialize=function(e){return JSON.stringify(e)};l.deserialize=function(e){if(typeof e!="string"){return undefined}return JSON.parse(e)};function b(){try{return(c in h&&h[c])}catch(e){return false}}function m(){try{return(n in h&&h[n]&&h[n][h.location.hostname])}catch(e){return false}}if(b()){g=h[c];l.set=function(e,o){if(o===undefined){return l.remove(e)}g.setItem(e,l.serialize(o))};l.get=function(e){return l.deserialize(g.getItem(e))};l.remove=function(e){g.removeItem(e)};l.clear=function(){g.clear()}}else{if(m()){g=h[n][h.location.hostname];l.set=function(e,o){if(o===undefined){return l.remove(e)}g[e]=l.serialize(o)};l.get=function(e){return l.deserialize(g[e]&&g[e].value)};l.remove=function(e){delete g[e]};l.clear=function(){for(var e in g){delete g[e]}}}else{if(k.documentElement.addBehavior){var j,f;try{f=new ActiveXObject("htmlfile");f.open();f.write('<script>document.w=window<\/script><iframe src="/favicon.ico"></frame>');f.close();j=f.w.frames[0].document;g=j.createElement("div")}catch(i){g=k.createElement("div");j=k.body}function a(e){return function(){var p=Array.prototype.slice.call(arguments,0);p.unshift(g);j.appendChild(g);g.addBehavior("#default#userData");g.load(c);var o=e.apply(l,p);j.removeChild(g);return o}}l.set=a(function(p,e,o){if(o===undefined){return l.remove(e)}p.setAttribute(e,l.serialize(o));p.save(c)});l.get=a(function(o,e){return l.deserialize(o.getAttribute(e))});l.remove=a(function(o,e){o.removeAttribute(e);o.save(c)});l.clear=a(function(q){var o=q.XMLDocument.documentElement.attributes;q.load(c);for(var p=0,e;e=o[p];p++){q.removeAttribute(e.name)}q.save(c)})}}}try{l.set(d,d);if(l.get(d)!=d){l.disabled=true}l.remove(d)}catch(i){l.disabled=true}if(typeof module!="undefined"){module.exports=l}else{if(typeof define==="function"&&define.amd){define(l)}else{this.store=l}}})();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.analytics.analytics-client:js-events', location = 'js/page-visibility.js' */
define("atlassian/analytics/page-visibility",function(){var a=(document.hidden!==undefined);var b={supported:a,isHidden:function(){return a?document.hidden:false}};return b});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.analytics.analytics-client:js-events', location = 'js/user-activity-xhr-header.js' */
define("atlassian/analytics/user-activity-xhr-header",["atlassian/analytics/page-visibility"],function(c){var e="x-atlassian-mau-ignore";var d=XMLHttpRequest.prototype.send;var a=window.fetch;var b=false;return{install:function(){if(!b&&c.supported){XMLHttpRequest.prototype.send=function(){if(c.isHidden()){this.setRequestHeader(e,c.isHidden())}d.apply(this,arguments)};if(a){window.fetch=function(f,i){var g=i||{};var h=g.headers;if(c.isHidden()){g.headers=(h)?new Headers(h):new Headers();g.headers.set(e,c.isHidden())}return a.call(this,f,g)}}b=true}},uninstall:function(){if(b){XMLHttpRequest.prototype.send=d;if(a){window.fetch=a}}b=false}}});require("atlassian/analytics/user-activity-xhr-header").install();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.analytics.analytics-client:js-events', location = 'js/atlassian-analytics.js' */
(function(d){var q=AJS.$.ajax;var l="atlassian-analytics";var j=typeof AJS.contextPath==="function"?AJS.contextPath():"";var p=null;var m=null;var g=null;var r="xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g,function(z){var y=Math.random()*16|0,x=z=="x"?y:(y&3|8);return x.toString(16)});var t=function(){var x="unknown";if(document.body.id=="jira"){x="jira"}else{if(document.body.id=="com-atlassian-confluence"){x="confluence"}}m=l+"."+x;g=m+".lock"};var f=function(){if(store.get(g)){return false}store.set(g,r);return(store.get(g)===r)};var u=function(){store.set(g,null)};var i=function(){var y=[],A,B,x,z;if(AJS.EventQueue.length==0){return}y=store.get(m)||y;for(x=0,z=AJS.EventQueue.length;x<z;++x){B=AJS.EventQueue[x];if(B.name){A={name:B.name,properties:B.properties,time:B.time||0};y.push(A)}}AJS.EventQueue.length=0;store.set(m,y)};var v=0;var a=function(x){return Math.min(5000*Math.pow(2,x),5*60*1000)};var e=0;var h=function(){var A;function y(){setTimeout(h,a(v=0))}function x(){setTimeout(h,a(++v))}if(!f()){++e;if(e<20){return y()}}else{e=0}try{i();A=store.get(m);if(!A||!A.length){return y()}store.remove(m);if(!o(A)){return y()}var B=new Date().getTime();for(var z=0;z<A.length;z++){if(A[z].time>0){A[z].timeDelta=A[z].time-B}else{A[z].timeDelta=z-A.length}delete A[z].time}p=q({type:"POST",url:j+"/rest/analytics/1.0/publish/bulk",data:JSON.stringify(A),contentType:"application/json",dataType:"json"});p.fail(function(){AJS.EventQueue.concat(A);i();x()});p.done(function(){y()})}finally{u()}};var o=function(z){for(var y=z.length-1;y>=0;y--){var B="";var A=z[y];var x=A.properties;if(typeof A.name==="undefined"){B="you must provide a name for the event."}else{if(typeof x!=="undefined"&&x!==null){if(x.constructor!==Object){B="properties must be an object with key value pairs."}else{b(x)}}}if(B!==""){AJS.log("WARN: Invalid analytics event detected and ignored, "+B+"\nEvent: "+JSON.stringify(A));z.splice(y,1)}}return z.length};var b=function(z){for(var y in z){if(z.hasOwnProperty(y)){var x=z[y];if(c(x)&&k(x)){}else{if(c(x)&&x.toString){z[y]=x.toString()}else{z[y]=""}}}}};function c(x){return typeof x!=="undefined"&&x!==null}function k(x){return typeof x==="number"||typeof x==="string"||typeof x==="boolean"}var n=function(){if(p&&!(p.state()==="resolved"||p.state()==="rejected")){p.abort()}};AJS.EventQueue=AJS.EventQueue||[];var s=Array.prototype.push;AJS.EventQueue.push=function(x){x.time=new Date().getTime();s.call(AJS.EventQueue,x)};AJS.toInit(function(){t();setTimeout(h,500);w()});d(window).on("beforeunload",function(){n();i()});AJS.Analytics={triggerPrivacyPolicySafeEvent:function(x,y){AJS.log("WARN: 'triggerPrivacyPolicySafeEvent' has been deprecated");AJS.EventQueue.push({name:x,properties:y})}};AJS.bind("analytics",function(x,y){AJS.EventQueue.push({name:y.name,properties:y.data})});AJS.bind("analyticsEvent",function(x,y){AJS.EventQueue.push({name:y.name,properties:y.data})});var w=function(){if(window.location.pathname.indexOf("/secure/admin/ViewApplicationProperties")>-1){AJS.$("[data-property-id='analytics-enabled']").remove()}else{if(window.location.pathname.indexOf("/secure/admin/EditApplicationProperties")>-1){var y=AJS.$(":contains(Enable Atlassian analytics)");if(y.size()>0){var x=y[y.size()-2];if(x){x.remove()}}}}}}(AJS.$));
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.analytics.analytics-client:policy-update-init', location = 'js/policy-update-init.js' */
AJS.toInit(function(){var a=WRM.data.claim("com.atlassian.analytics.analytics-client:policy-update-init.policy-update-data-provider");if(a){WRM.require("wrc!com.atlassian.analytics.analytics-client:policy-update")}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.analytics.analytics-client:programmatic-analytics-init', location = 'js/programmatic-analytics-init.js' */
(function(){var a=WRM.data.claim("com.atlassian.analytics.analytics-client:programmatic-analytics-init.programmatic-analytics-data-provider");if(a){WRM.require("wrc!com.atlassian.analytics.analytics-client:programmatic-analytics")}}());
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:application-header-administration-cog-resource', location = 'header/cog.js' */
var NavLinks=(function(a){a.ApplicationHeader=function(b){b.Cog=(function(){var c=function(){var d=AJS.$("#system-admin-menu-content");if(d.length>0){return d}var e=AJS.$("#admin-menu-link-content");if(e.length>0){return e}return AJS.$("#bamboo\\.global\\.header-admin\\.menu")};return{getDropdown:c}}());return b}(a.ApplicationHeader||{});return a}(NavLinks||{}));
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:administration-shortcuts-resources', location = 'adminshortcuts/adminshortcuts.soy' */
// This file was automatically generated from adminshortcuts.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace navlinks.templates.adminshortcuts.
 */

if (typeof navlinks == 'undefined') { var navlinks = {}; }
if (typeof navlinks.templates == 'undefined') { navlinks.templates = {}; }
if (typeof navlinks.templates.adminshortcuts == 'undefined') { navlinks.templates.adminshortcuts = {}; }


navlinks.templates.adminshortcuts.section = function(opt_data, opt_ignored) {
  var param5 = '<ul class="aui-list-truncate">';
  var linkList7 = opt_data.links;
  var linkListLen7 = linkList7.length;
  for (var linkIndex7 = 0; linkIndex7 < linkListLen7; linkIndex7++) {
    var linkData7 = linkList7[linkIndex7];
    param5 += '<li><a href="' + soy.$$escapeHtml(linkData7.link) + '">' + soy.$$escapeHtml(linkData7.label) + '</a></li>';
  }
  param5 += '</ul>';
  var output = '' + aui.dropdown2.section({id: 'nl-remote-admin-section', label: 'Other applications', content: param5});
  return output;
};
if (goog.DEBUG) {
  navlinks.templates.adminshortcuts.section.soyTemplateName = 'navlinks.templates.adminshortcuts.section';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:administration-shortcuts-resources', location = 'adminshortcuts/adminnavlinks.js' */
var NavLinks=(function(a){a.AdminShortcuts=(function(){var c=function(){return AJS.$.ajax({url:AJS.contextPath()+"/rest/menu/latest/admin",cache:false,dataType:"json"})};var b=function(){AJS.$("#nl-remote-admin-section").on("click","a",function(){NL.trackEvent("remoteAdminItemSelected",NL.getCurrentApplication(),$(this).attr("href"))})};return{render:function(){c().done(function(e){e=_.reject(e,function(f){return f.local===true});if(e.length){var d=navlinks.templates.adminshortcuts.section({links:e});a.ApplicationHeader.Cog.getDropdown().append(d);b()}})}}}());return a}(NavLinks||{}));
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:administration-shortcuts', location = 'adminshortcuts/adminshortcuts-cdn.js' */
AJS.toInit(function(){if(AJS.DarkFeatures&&AJS.DarkFeatures.isEnabled("rotp.admin.shortcuts")){NavLinks.AdminShortcuts.render()}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.splitchunk.vendors--02a4084a95', location = 'aui.chunk.9835e22e1b78c86dc53f--2123f72bb06ff40ad0fe.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.splitchunk.vendors--02a4084a95"],{M4tZ:function(E,n,o){var i,u,A;
/*!
 * jQuery UI Keycode 1.12.1
 * http://jqueryui.com
 *
 * Copyright jQuery Foundation and other contributors
 * Released under the MIT license.
 * http://jquery.org/license
 */u=[o("oDIA"),o("O+lX")],void 0===(A="function"==typeof(i=function(E){return E.ui.keyCode={BACKSPACE:8,COMMA:188,DELETE:46,DOWN:40,END:35,ENTER:13,ESCAPE:27,HOME:36,LEFT:37,PAGE_DOWN:34,PAGE_UP:33,PERIOD:190,RIGHT:39,SPACE:32,TAB:9,UP:38}})?i.apply(n,u):i)||(E.exports=A)}}]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_jquery.ui.keycode', location = 'aui.chunk.09b64181270e2539dc95--ac939fece05e0209a225.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["jquery.ui.keycode"],{wrJm:function(u,i,n){"use strict";n.r(i);n("M4tZ");i.default="jquery"}},[["wrJm","runtime","aui.splitchunk.vendors--20a97d6a33","aui.splitchunk.vendors--02a4084a95","aui.splitchunk.0d131bcbf1"]]]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:rotp-projectshortcuts', location = 'projectshortcuts/projectshortcuts.soy' */
// This file was automatically generated from projectshortcuts.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace navlinks.templates.projectshortcuts.
 */

if (typeof navlinks == 'undefined') { var navlinks = {}; }
if (typeof navlinks.templates == 'undefined') { navlinks.templates = {}; }
if (typeof navlinks.templates.projectshortcuts == 'undefined') { navlinks.templates.projectshortcuts = {}; }


navlinks.templates.projectshortcuts.dialogContent = function(opt_data, opt_ignored) {
  return '' + ((opt_data.localShortcuts && opt_data.localShortcuts.length > 0) ? navlinks.templates.projectshortcuts.dialogContentShortcuts({shortcuts: opt_data.localShortcuts, listClass: 'projectshortcut-links'}) : '') + ((opt_data.remoteShortcuts != null) ? (opt_data.remoteShortcuts.length > 0) ? '<h2 class="projectshortcuts-heading">Related Links</h2>' + navlinks.templates.projectshortcuts.dialogContentShortcuts(soy.$$augmentMap(opt_data.remoteShortcuts, {shortcuts: opt_data.remoteShortcuts, listClass: 'projectshortcut-links'})) : '' : navlinks.templates.projectshortcuts.dialogLoading(null));
};
if (goog.DEBUG) {
  navlinks.templates.projectshortcuts.dialogContent.soyTemplateName = 'navlinks.templates.projectshortcuts.dialogContent';
}


navlinks.templates.projectshortcuts.headingWrapper = function(opt_data, opt_ignored) {
  return '<div class="project-dialog-header-wrapper"><div class="project-header"><img class="project-img" src="' + soy.$$escapeHtml(opt_data.logoUrl) + '"><h2 class="dialog-title">' + soy.$$escapeHtml(opt_data.title) + '</h2></div><div class="project-content-wrapper">' + soy.$$filterNoAutoescape(opt_data.contentHtml) + '</div></div>';
};
if (goog.DEBUG) {
  navlinks.templates.projectshortcuts.headingWrapper.soyTemplateName = 'navlinks.templates.projectshortcuts.headingWrapper';
}


navlinks.templates.projectshortcuts.dialogContentShortcuts = function(opt_data, opt_ignored) {
  opt_data = opt_data || {};
  var output = '<ul' + ((opt_data.listClass) ? ' class="' + soy.$$escapeHtml(opt_data.listClass) + '"' : '') + '>';
  var shortcutList35 = opt_data.shortcuts;
  var shortcutListLen35 = shortcutList35.length;
  for (var shortcutIndex35 = 0; shortcutIndex35 < shortcutListLen35; shortcutIndex35++) {
    var shortcutData35 = shortcutList35[shortcutIndex35];
    output += '<li' + ((shortcutIndex35 == shortcutListLen35 - 1) ? ' class="last"' : '') + '>' + navlinks.templates.projectshortcuts.dialogContentShortcut(shortcutData35) + '</li>';
  }
  output += '</ul>';
  return output;
};
if (goog.DEBUG) {
  navlinks.templates.projectshortcuts.dialogContentShortcuts.soyTemplateName = 'navlinks.templates.projectshortcuts.dialogContentShortcuts';
}


navlinks.templates.projectshortcuts.dialogContentShortcut = function(opt_data, opt_ignored) {
  return '<a href="' + soy.$$escapeHtml(opt_data.link) + '"' + ((opt_data.tooltip) ? ' title="' + soy.$$escapeHtml(opt_data.tooltip) + '"' : '') + '>' + soy.$$escapeHtml(opt_data.label) + '</a>';
};
if (goog.DEBUG) {
  navlinks.templates.projectshortcuts.dialogContentShortcut.soyTemplateName = 'navlinks.templates.projectshortcuts.dialogContentShortcut';
}


navlinks.templates.projectshortcuts.dialogLoading = function(opt_data, opt_ignored) {
  opt_data = opt_data || {};
  return '<div class="projectshortcuts-loading">' + ((opt_data.text) ? soy.$$escapeHtml(opt_data.text) : '') + '</div>';
};
if (goog.DEBUG) {
  navlinks.templates.projectshortcuts.dialogLoading.soyTemplateName = 'navlinks.templates.projectshortcuts.dialogLoading';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:rotp-projectshortcuts', location = 'projectshortcuts/projectshortcuts.js' */
(function(e,g){var i,k={},m="key",b="name",j="entity-type";function f(s){var n=e(this),o=n.data(m),q=n.data(b),p=n.data(j);if(typeof o==="undefined"){return}s.preventDefault();i=new AJS.Dialog({width:600,keypressListener:function(u){if(u.which==jQuery.ui.keyCode.ESCAPE){i.remove()}},id:"project-shortcuts-dialog"}).addCancel("Close",function(){i.remove()}).addPanel("",navlinks.templates.projectshortcuts.headingWrapper({title:q,logoUrl:h(),contentHtml:navlinks.templates.projectshortcuts.dialogLoading({text:"Retrieving links…"})})).show();c(i);if(!k[o]){k[o]={entity:{title:q},localShortcuts:null,remoteShortcuts:null};d(AJS.contextPath()+"/rest/project-shortcuts/1.0/local/"+o,{entityType:p}).done(t);d(AJS.contextPath()+"/rest/project-shortcuts/1.0/remote/"+o,{entityType:p}).done(r).fail(function(){var u=i.getCurrentPanel().body.find(".project-content-wrapper");u.find(".projectshortcuts-loading").remove();AJS.messages.error(u,{body:"Could not retrieve remote project shortcuts",closeable:false});c(i)})}else{l(k[o])}function t(u){k[o].localShortcuts=u.shortcuts;l(k[o])}function r(u){k[o].remoteShortcuts=u.shortcuts;l(k[o])}}function h(){return e(".project-shortcut-dialog-trigger img").attr("src")}function l(n){if(n.localShortcuts){i.getCurrentPanel().html(navlinks.templates.projectshortcuts.headingWrapper({title:n.entity.title,logoUrl:h(),contentHtml:navlinks.templates.projectshortcuts.dialogContent(n)}));c(i)}}function a(p){var o=210;if(!p||p.length<=o){return p}var n=o;while(n>0&&p.charAt(n)!=" "){n--}if(n==0){n=o}p=p.substring(0,n);if(p.length>=n){p=p+"..."}return p}function c(n){var q=n.popup.element,p=q.find(".dialog-panel-body"),o=q.find(".dialog-components");p.height("auto");q.height(o.outerHeight()-1);e(".aui-shadow").remove()}function d(n,o){return e.ajax({url:n,cache:false,data:o,dataType:"json"})}e(document).on("click",".project-shortcut-dialog-trigger",f)}(jQuery,window.NL=(window.NL||{})));
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:atlassian-ui-popup-display-controller', location = 'popups/DisplayController.js' */
AJS.Popups=AJS.Popups||{};AJS.Popups.DisplayController=function(){var c=[];var a=false;var b=false;AJS.toInit(function(){setTimeout(function(){AJS.Popups.DisplayController.render()},0)});return{request:function(d){c.push(d);if(a&&b===false){this.render()}},render:function(){c.sort(function(e,d){return e.weight-d.weight});a=true;if(c.length!==0){b=true;c[0].show()}}}}();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.page.iconography', location = 'aui.chunk.0e489f4b2b9c5e735d98--50e9ccfcf1ce3886e7d4.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.page.iconography"],[],[["rSV2","runtime","aui.splitchunk.fbbef27525","aui.splitchunk.d7c46c2734","aui.splitchunk.56dfb54d0c","aui.splitchunk.908fe798b4"]]]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.extra.jira:jira-issues-view-mode-resources', location = 'jira/jira-issues-view-mode/main.js' */
require(["ajs","jquery"],function(a,d){var c=function(){var b=d(".wiki-content [data-jira-key][data-client-id]");if(0==b.length)return!1;WRM.require("wr!confluence.extra.jira:jira-issues-view-mode-async-resource",function(){require(["confluence/jim/jira/jira-issues-view-mode/lazy-loading","confluence/jim/jira/jira-issues-view-mode/fix-ui"],function(e,f){e.init(b).done(function(){f.fixBreakIconInOldConf()})})})};a.toInit(c);a.bind("ic-jim-async-supported",c)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.extra.jira:amd-support', location = 'amd/confluence-shim.js' */
define("confluence/jim/confluence-shim",function(){return window.Confluence});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.extra.jira:amd-support', location = 'amd/amd-exporter.js' */
define("confluence/jim/amd/module-exporter",[],function(){var f={namespace:function(d,e,a){var b=d.split(".");a=a||window;var c,h=b.length-1;for(c=0;c<h;c++){var g=a[b[c]];a=null!=g?g:a[b[c]]={}}a[b[c]]&&window.console&&window.console.warn&&window.console.warn('Value of "'+d+'" was overridden');a[b[c]]=e||a[b[c]]||{};return a[b[c]]},safeRequire:function(d,e){define&&void 0===define.amd&&(d=require(d),e&&e(d))},exportModuleAsGlobal:function(d,e,a){f.safeRequire(d,function(b){f.namespace(e,b);a&&a(b)})}};
return f});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:lodash-amd', location = 'applinks/internal/lib/lodash-amd-global.js' */
define('applinks/lib/lodash', function() {
        if (!window._) {
            throw "lodash not found";
        }
        return window._;
    });
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/lib-version.js' */
// NOTE: this is in "lib" because it's required by aui-amd.js. This module _cannot_ have any dependencies on non-lib
// modules (like "common")
define('applinks/lib/version', [
    'applinks/lib/lodash'
], function(
    _
) {

    function checkIntValue(value, desc) {
        if (_.isUndefined(value)) {
            return 0;
        } else if (typeof value !== 'number' || isNaN(value)) {
            throw new Error(desc + ': expected a number, was: <' + value + '>');
        } else {
            return Math.floor(value);
        }
    }

    function checkVersionObject(object, desc) {
        if (!object || !(object instanceof Version)) {
            throw new Error(desc + ': expected a Version object, was: <' + object + '>');
        }
        return object;
    }

    function compareInt(intOne, intTwo) {
        return intOne > intTwo ? 1 : intOne == intTwo ? 0 : -1;
    }

    /**
     * Constructs a version object that contains version information and can be compared with other version objects.
     *
     * @param major major version component
     * @param minor minor version component
     * @param bugfix bugfix version component
     * @constructor
     */
    function Version(major, minor, bugfix) {
        this.major = checkIntValue(major, 'major');
        this.minor = checkIntValue(minor, 'minor');
        this.bugfix = checkIntValue(bugfix, 'bugfix');
    }

    /**
     * Constructs a Version by parsing a version string.
     *
     * @param versionString version string to parse, expected to be in the form of <major>.<minor>.<bugfix>
     * @param versionDesc what version does the `versionString` represent
     */
    Version.parse = function(versionString, versionDesc) {
        versionDesc = versionDesc || 'versionString';
        if (!versionString) {
            throw new Error(versionDesc + ': expected a non-empty string');
        }
        var versionSplit = versionString.split('.');
        if (versionSplit.length !== 3) {
            throw new Error(versionDesc + ': expected <major>.<minor>.<bugfix> string, was: <' + versionString + '>');
        }
        return new Version(parseInt(versionSplit[0]), parseInt(versionSplit[1]), parseInt(versionSplit[2]));
    };

    /**
     * Comparator function to sort Version objects.
     *
     * @param versionOne first version object
     * @param versionTwo second version object
     * @returns {Number} comparison result
     * @see Version.compareTo(that)
     */
    Version.comparator = function(versionOne, versionTwo) {
        return checkVersionObject(versionOne, 'versionOne').compareTo(checkVersionObject(versionTwo, 'versionTwo'));
    };

    /**
     * Return -1, 0 or 1 as this Version object represents a version less than, equal to, or greater than `that`.
     *
     * @param that the other version to compare to
     * @returns {Number} comparison result
     */
    Version.prototype.compareTo = function(that) {
        checkVersionObject(that, 'that');
        var majorResult = compareInt(this.major, that.major);
        if (majorResult != 0) {
            return majorResult;
        }
        var minorResult = compareInt(this.minor, that.minor);
        if (minorResult != 0) {
            return minorResult;
        }
        return compareInt(this.bugfix, that.bugfix);
    };

    Version.prototype.greaterThan = function(that) {
        return this.compareTo(that) > 0;
    };

    Version.prototype.greaterThanOrEqual = function(that) {
        return this.compareTo(that) >= 0;
    };

    Version.prototype.lessThan = function(that) {
        return this.compareTo(that) < 0;
    };

    Version.prototype.lessThanOrEqual = function(that) {
        return this.compareTo(that) <= 0;
    };

    Version.prototype.equals = function(that) {
        return this.compareTo(that) == 0;
    };

    Version.prototype.toMinor = function() {
        return new Version(this.major, this.minor);
    };

    Version.prototype.toMajor = function() {
        return new Version(this.major);
    };

    Version.prototype.toString = function() {
        return this.major + '.' + this.minor + '.' + this.bugfix;
    };

    return Version;
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/aui-version-details.js' */
define('applinks/lib/aui-version-details', [
    'applinks/lib/version'
], function(
    Version
) {
    // minimum 5.9.x AUI versions - lower versions ship with bugs that break some of the Applinks functionality.
    var MINIMUM_59_VERSION = new Version(5, 9, 13);
    var VERSION_58 = new Version(5, 8);
    var VERSION_59 = new Version(5, 9);

    function checkVersion(version) {
        // if 5.9, must be >= the minimum required 5.9.x version
        if (version.lessThan(MINIMUM_59_VERSION) || version.lessThan(VERSION_59)) {
            throw new Error('AUI version ' + version + ' is too low, you need to upgrade AUI to ' + MINIMUM_59_VERSION
                + ' for Applinks to work');
        }
    }

    function addVersionDetails(AJS) {
        var ajsVersion = Version.parse(AJS.version, 'AUI version');
        checkVersion(ajsVersion);

        AJS.versionDetails = ajsVersion;
        AJS.versionDetails.is58 = ajsVersion.toMinor().equals(VERSION_58);
        AJS.versionDetails.is59 = ajsVersion.toMinor().equals(VERSION_59);

        return AJS;
    }

    return  {
        /**
         * Check AUI version for compatibility with Applinks and add `versionDetails` field.
         */
        addVersionDetails: addVersionDetails
    };
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/console-amd.js' */
define('applinks/lib/console', function() {
    return window.console;
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/jquery-amd.js' */
define('applinks/lib/jquery', function() {
   return window.jQuery;
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/aui-amd.js' */
define('applinks/lib/aui', [
    'applinks/lib/window',
    'applinks/lib/aui-version-details'
], function(
    window,
    VersionDetails
) {
    var AJS = window.AJS;
    if (!AJS) {
        throw new Error('window.AJS not defined, cannot load AUI');
    }

    return VersionDetails.addVersionDetails(AJS);
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/wrm-amd.js' */
/**
 * Define Web resource manager as an AMD dependency. This should also be present in apps using Atlassian Plugins Web
 * Resources framework.
 */
define('applinks/lib/wrm', function() {
   return window.WRM;
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/skate-amd.js' */
define('applinks/lib/skate', function() {
   // window.skate is not exposed as a global from AUI 5.9, this will return undefined with that version or later
   return window.skate;
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/window-amd.js' */
/**
 * Define window as AMD module to facilitate unit testing of some modules
 */
define('applinks/lib/window', function() {
    return window;
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/spi-amd.js' */
/**
 * Define the Applinks SPI functions as AMD module to facilitate unit testing of some modules
 */
define('applinks/lib/spi', function() {
    return {get:
        function (){
            //required since code that depends on this global is run before the global is defined,
            // can be eliminated once this module is made properly AMD
            return AppLinks.SPI;
    }}
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-lib', location = 'applinks/internal/lib/applinks-creation-amd.js' */
/**
 * Define the Applinks creation resources  as AMD module to facilitate unit testing of some modules
 */
define('applinks/lib/creation', function() {
    return {
        get: function () {
            //required since code that depends on this global is run before the global is defined,
            // can be eliminated once this module is made properly AMD
            return AppLinks.Creation;
        }
    }
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-common-exported', location = 'applinks/internal/common/docs.js' */
define('applinks/common/docs', [
    'applinks/lib/jquery',
    'applinks/lib/aui',
    'applinks/common/help-paths'
], function(
    $,
    AJS,
    ApplinksHelpPaths
) {
    // NOTE: should be moved to applinks/feature/help-link, see APLDEV-593

    return {
        /**
         * NOTE: this is a dynamically generated version of the link build in _help_link.vm, any update here should be
         * applied there.
         * @method createDocLink
         * @param pageKey a key that maps to a page in ual-help-paths.properties
         * @param sectionKey (Optional) a key that maps to an anchor section id in ual-help-paths.properties
         * @param classNames (Optional) Whitespace separated list of additional class names
         * @return an html &lt;a&gt; element targeting the specified page & section
         */
        createDocLink: function(pageKey, sectionKey, classNames) {
            return this._createDocLink(pageKey, sectionKey, classNames, "Help");
        },
        createDocLinkIcon: function(pageKey, sectionKey, classNames) {
            return this._createDocLink(pageKey, sectionKey, classNames, '')
                .append($('<span/>', {
                    "class": 'aui-icon aui-icon-small aui-iconfont-help',
                    "text": "Help"
                }));
        },

        _createDocLink: function(pageKey, sectionKey, classNames, text) {
            if (!classNames) {
                classNames = '';
            } else {
                classNames = ' ' + classNames;
            }
            return $('<a/>', {
                'class': 'ual-help-link help-link' + classNames,
                href: this.getDocHref(pageKey, sectionKey),
                target: '_blank',
                'data-help-link-key': pageKey,
                text: text,
                title: "Help"
            });
        },

        /**
         * @method getDocHref
         * @param pageKey a key that maps to a page in ual-help-paths.properties
         * @param sectionKey (Optional) a key that maps to an anchor section id in ual-help-paths.properties
         * @return the url of the given page and section (if specified)
         */
        getDocHref: function(pageKey, sectionKey) {
            var link = ApplinksHelpPaths.getFullPath(pageKey);
            if (sectionKey) {
                link += '#' + ApplinksHelpPaths.getPath(sectionKey);
            }
            return link;
        }
    };

});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-common-exported', location = 'applinks/internal/common/help-paths.js' */
define('applinks/common/help-paths', [
    'applinks/lib/console',
    'applinks/lib/wrm',
    'applinks/lib/lodash',
    'applinks/common/modules',
    'applinks/common/preconditions'
], function(
    console,
    WRM,
    _,
    ApplinksModules,
    Preconditions
) {
    // NOTE: should be moved to applinks/feature/help-link, see APLDEV-593

    // lazy-load help paths, facilitates unit-testing
    var allHelpPaths = _.memoize(function() {
        var helpPaths = WRM.data.claim(ApplinksModules.dataFqn(ApplinksModules.COMMON_EXPORTED, 'applinks-help-paths'));
        if (!helpPaths.entries) {
            console.warn('Help paths not found, all help links are likely to be broken.');
        }
        return helpPaths.entries || {};
    });

    var getPath = function(key, sectionKey) {
        Preconditions.nonEmptyString(key, 'key');
        var path = allHelpPaths()[key] || key;
        if (sectionKey) {
            Preconditions.nonEmptyString(sectionKey, 'sectionKey');
            var prefix = path.replace(/\+/g, ''); // "g" flag to remove _all_ '+' signs
            path += '#' +prefix + '-' + sectionKey;
        }
        return path;
    };

    function endsWith(string, suffix) {
        return string.indexOf(suffix, string.length - suffix.length) !== -1;
    }

    function addSuffixIfRequired(string, suffix) {
        return endsWith(string, suffix) ? string : string + suffix;
    }

    return {
        /**
         * @param key {string} key to get the path for
         * @returns {string} relative help path that can be appended to any relevant docs base URL
         */
        getPath: getPath,

        /**
         * @param key {string} key to get the path for
         * @param sectionKey {string} optional key of the anchor on the target page
         * @returns {string} full help path including the base URL
         */
        getFullPath: function(key, sectionKey) {
            var baseUrl = this.baseUrl();
            return addSuffixIfRequired(baseUrl, '/') + this.getPath(key, sectionKey);
        },

        /**
         * @returns {string} configured base URL for the help paths
         */
        baseUrl: _.partial(getPath, 'applinks.docs.root')
    }
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-common-exported', location = 'applinks/internal/common/events.js' */
define('applinks/common/events', [
    'applinks/lib/jquery',
    'applinks/lib/lodash',
    'applinks/lib/window',
    'applinks/common/preconditions'
], function(
    $,
    _,
    window,
    Preconditions
) {
    var PREFIX = 'applinks.event.';

    function applinksEvent(eventId) {
        return PREFIX + Preconditions.nonEmptyString(eventId, 'eventId');
    }

    /**
     * Provides common Applinks event IDs and a simple event system facade API. This is a preferred way to subscribe to
     * and raise Applinks-specific events as it does not depend on a specific event bus or event target (such as
     * `document`), as well as facilitates unit testing.
     */
    return {
        PREREADY: applinksEvent('preready'),
        READY: applinksEvent('ready'),

        /**
         * Raised when applinks list is first loaded
         */
        APPLINKS_LOADED: applinksEvent('loaded'),
        /**
         * Raised when applinks list is updated
         */
        APPLINKS_UPDATED: applinksEvent('updated'),

        /**
         * This event is only raised when linking to Atlassian applications
         * Can be consumed by other plugins
         */
        NEW_APPLINK_CREATED: applinksEvent('created'),

        /**
         * Raised when orphaned upgrade operation succeeds in creating a new Applink from the orphaned relationship
         */
        ORPHANED_UPGRADE: applinksEvent('orphaned.upgrade'),

        /**
         * Raised when v3 onboarding has finished or, or has never run on the current page (and won't).
         */
        V3_ONBOARDING_FINISHED: applinksEvent('v3-onboarding-finished'),

        // legacy events
        Legacy: {
            MESSAGE_BOX_DISPLAYED: applinksEvent('message-box-displayed')
        },

        applinksEvent: applinksEvent,

        on: function(events, handler, context) {
            var handlerWithContext = context ? _.bind(handler, context) : handler;
            $(window.document).on(events, handlerWithContext);
        },

        off: function(events, handler) {
            $(window.document).off(events, handler);
        },

        trigger: function(event, data) {
            $(window.document).trigger(event, data);
        }
    }
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-common-exported', location = 'applinks/internal/common/i18n.js' */
define('applinks/common/i18n', [
    'applinks/lib/lodash',
    'applinks/lib/jquery',
    'applinks/lib/wrm',
    'applinks/common/modules',
    'applinks/common/preconditions',
    'applinks/common/products'
], function(
    _,
    $,
    WRM,
    ApplinksModules,
    Preconditions,
    ApplinksProducts
) {
    var getAllEntityTypes = _.memoize(function() {
        var val = WRM.data.claim(ApplinksModules.dataFqn(ApplinksModules.COMMON_EXPORTED, 'entity-types'));
        return Preconditions.hasValue(val, 'entity-types', 'Entity Types data not found');
    });

    var getAllAuthTypes = _.memoize(function() {
        var val = WRM.data.claim(ApplinksModules.dataFqn(ApplinksModules.COMMON_EXPORTED, 'authentication-types'));
        return Preconditions.hasValue(val, 'authentication-types', 'Authentication Types data not found');
    });

    return {
        
        /**
         * @param typeId ID of the application type to resolve
         * @returns {string} resolved i18n-ed type name, or the original `typeId` if there is no mapping
         */
        getApplicationTypeName: function(typeId) {
            return ApplinksProducts.getTypeName(typeId);
        },

        /**
         * @param typeId ID of the entity type to resolve
         * @returns {string} resolved i18n-ed singular entity type name, or the original `typeId` if there is no mapping
         */
        getEntityTypeName: function(typeId) {
            return getAllEntityTypes().singular[typeId] || typeId;
        },

        /**
         * @param typeId ID of the entity type to resolve
         * @returns {string} resolved i18n-ed plural entity type name, or the original `typeId` if there is no mapping
         */
        getPluralizedEntityTypeName: function(typeId) {
            return getAllEntityTypes().plural[typeId] || typeId;
        },

        /**
         * @param type ID of the authentication type to resolve (usually in a form of full class name)
         * @returns {string} resolved i18n-ed authentication type name, or the original `type` if there is no mapping
         */
        getAuthenticationTypeName: function(type) {
            return getAllAuthTypes()[type] || type;
        }
    };
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-common-exported', location = 'applinks/internal/common/modules.js' */
/**
 * Applinks plugin modules core constants and definitions.
 */
define('applinks/common/modules', function() {
    return {
        /**
         * Applinks plugin key
         */
        PLUGIN_KEY: 'com.atlassian.applinks.applinks-plugin',

        // key web resource keys
        COMMON_EXPORTED: 'applinks-common-exported',
        COMMON: 'applinks-common',

        /**
         * Fully qualifies a module name using the plugin key.
         *
         * @param {string} moduleName module name to qualify
         * @returns {string} fully qualified name
         */
        fqn: function(moduleName) {
            return this.PLUGIN_KEY + ':' + moduleName;
        },

        /**
         * Fully qualifies web-resource data using module name and data key.
         *
         * @param {string} moduleName module name
         * @param {string} dataKey key of the data element
         * @returns {string} fully qualified name
         */
        dataFqn: function(moduleName, dataKey) {
            return this.fqn(moduleName) + '.' + dataKey;
        }
    };
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-common-exported', location = 'applinks/internal/common/preconditions.js' */
define('applinks/common/preconditions', [
    'applinks/lib/lodash'
], function(
    _
) {
    function nonEmptyString(value, varName, customMessage) {
        return _checkArgument(
            _.isString(value) && !_.isEmpty(value),
            customMessage,
            _withVarName(varName, ': expected a non-empty string, was: <' + value + '>'),
            value
        );
    }

    function isFunction(value, varName, customMessage) {
        return _checkArgument(
            _.isFunction(value),
            customMessage,
            _withVarName(varName, ': expected a function, was: ' + value),
            value
        );
    }

    function isArray(value, varName, customMessage) {
        return _checkArgument(
            _.isArray(value),
            customMessage,
            _withVarName(varName, ': expected an array, was: ' + value),
            value
        );
    }

    function hasValue(value, varName, customMessage) {
        return _checkArgument(
            value,
            customMessage,
            _withVarName(varName, ': expected a value'),
            value
        );
    }

    function _checkArgument(test, message, defaultMessage, actualValue) {
        var actualMessage = message ? message : defaultMessage;
        if (!test) {
            throw new Error(actualMessage)
        }
        return actualValue || test;
    }

    function _withVarName(varName, msg) {
        return (varName || '[unspecified]') + msg;
    }

    return {
        // applinks support 1.5.2+ underscore and its lodash counterparts
        // _.partial works differently in lodash@2.x vs its newer and underscore versions
        //  thus replacing _.partial with vanilla JS for full compatibility
        checkArgument: function(test, message, actualValue) {
            return _checkArgument(test, message, '', actualValue);
        },
        nonEmptyString: nonEmptyString,
        isArray: isArray,
        isFunction: isFunction,
        hasValue: hasValue
    };
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-common-exported', location = 'applinks/internal/common/products.js' */
define('applinks/common/products', [
    'applinks/lib/lodash',
    'applinks/lib/wrm',
    'applinks/common/modules',
    'applinks/common/preconditions'
], function(
    _,
    WRM,
    ApplinksModules,
    Preconditions
) {
    var getAllTypes = _.memoize(function() {
        var val = WRM.data.claim(ApplinksModules.dataFqn(ApplinksModules.COMMON_EXPORTED, 'applinks-types'));
        return Preconditions.hasValue(val, 'types', 'Application Types data not found');
    });

    /**
     * @param typeId ID of the application type to resolve
     * @returns {string} resolved i18n-ed type name, or the original `typeId` if there is no mapping
     */
    function getTypeName(typeId) {
        return getAllTypes()[typeId] || typeId;
    }

    /**
     * Map of Atlassian product keys to application type IDs
     */
    return {
        BAMBOO: 'bamboo',
        BITBUCKET: 'stash', // special case, see java class com.atlassian.applinks.application.bitbucket.BitbucketApplicationTypeImpl.TYPE_ID
        CONFLUENCE: 'confluence',
        FECRU: 'fecru',
        JIRA: 'jira',
        REFAPP: 'refapp',
        STASH: 'stash',
        getTypeName: getTypeName
    };
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.auiplugin:split_aui.core', location = 'aui.chunk.81859c77a35803f186d6--a965583ff2f1059e0380.js' */
(window.__auiJsonp=window.__auiJsonp||[]).push([["aui.core"],[],[["7KGX","runtime","aui.splitchunk.0d131bcbf1","aui.splitchunk.444efc83be","aui.splitchunk.739b9ec8cc","aui.splitchunk.994e478d48","aui.splitchunk.479fe6ee76","aui.splitchunk.2e16019fb9","aui.splitchunk.8771ceac91","aui.splitchunk.7873ba9060","aui.splitchunk.0fc208f3fe","aui.splitchunk.2a34183e72"]]]);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-util-js', location = 'applinks/internal/non-amd/rest-service.js' */
// NOTE: this is used outside of Applinks. See atlassian-plugin.xml for more details about the associated restrictions

(function(AppLinksI18n, ApplinksDocs, ApplinksEvents) {
    var $ = AJS.$;

    /**
     * The triggering of AppLinks initialisation can be customised by setting a function on
     * AJS.AppLinksInitialisationBinder. The binder function should take a single argument which is a zero-arg function to
     * run and should execute this function when appropriate.
     */
    AppLinks = $.extend(window.AppLinks || {}, {
        Event: {
            NAMESPACE: 'applinks'
        },
        I18n: AppLinksI18n,
        Docs: ApplinksDocs
    });
    AppLinks.Event = $.extend(window.AppLinks.Event, ApplinksEvents);

    // Is there an overridden initialisation binder?
    if (AJS.AppLinksInitialisationBinder) {
        AppLinks.initialisationBinder = AJS.AppLinksInitialisationBinder;
    } else {
        // The default bind if no specific binder is specified
        AppLinks.initialisationBinder = function(f) {
            AJS.toInit(f);
        }
    }

    function parseJsonResponse(xhr) {
        var respJSON = xhr.responseText;
        var respObj;
        try {
            respObj = JSON.parse(respJSON);
        } catch (e) {
            console && console.error && console.error('invalid JSON response', respJSON, xhr);
        }
        return respObj || {};
    }

    var restUrlVersionMatch = new RegExp('rest/applinks(?:.*?)/(\\d\\.\\d|\\d)/');
    /**
     * Determine the REST endpoint version of a given URL.
     * @param url
     */
    function versionOf(url) {
        var results = restUrlVersionMatch.exec(url);
        return results && results.length === 2 ? results[1] : false;
    }

    AppLinks.initialisationBinder(function() {
        AppLinks = $.extend(window.AppLinks || {}, {
            failure: function(data) {
                if (data.status == 401) {
                    window.location.reload();
                } else {
                    var message = AppLinks.parseError(data);
                    var errorDivs = $('.page-error');

                    if (errorDivs.length > 0) {
                        errorDivs.html(message).fadeIn('slow');
                    } else {
                        alert('REST request failed: ' + message);
                    }
                }
            },
            jsonRequest: function(url, type, data, success, error, beforeSend) {
                if (data) {
                    data = JSON.stringify(data);
                }
                $(".page-error").fadeOut('fast');
                if (!error) error = AppLinks.failure;
                return jQuery.ajax({
                    url: url,
                    type: type,
                    data: data,
                    dataType: 'json',
                    contentType: "application/json; charset=utf-8",
                    beforeSend: beforeSend,
                    cache: false,
                    success: success,
                    error: error,
                    jsonp: false
                });
            },
            xmlRequest: function(url, type, data, success, error, beforeSend) {
                if (data) {
                    data = JSON.stringify(data);
                }
                $(".page-error").fadeOut('fast');
                if (!error) error = AppLinks.failure;
                return jQuery.ajax({
                    url: url,
                    type: type,
                    data: data,
                    dataType: 'xml',
                    contentType: "application/xml; charset=utf-8",
                    beforeSend: beforeSend,
                    cache: false,
                    success: success,
                    error: error
                });
            },
            parseError: function(errorData) {
                var error = parseJsonResponse(errorData);
                var message = error && error.message;
                if (message) {
                    return $.isArray(message) ? message.join(' ') : message;
                }
                if (errorData) {
                    return errorData.statusText ? errorData.statusText : errorData;
                }
                return 'An unknown error occurred. Check the console logs for details.';
            },
            put: function(url, data, success, error, beforeSend) {
                return AppLinks.jsonRequest(url, 'PUT', data, success, error, beforeSend);
            },
            post: function(url, data, success, error, beforeSend) {
                return AppLinks.jsonRequest(url, 'POST', data, success, error, beforeSend);
            },
            update: function(data, success, error, beforeSend) {
                var selfLink = AppLinks.self_link(data);
                /**
                 * Adapt to the version of the endpoint and the HTTP verb it uses.
                 * yes, this is not "REST-ful". The verb is wrong. It has been wrong since 2014.
                 * It is simpler to adapt than it is to release a new endpoint.
                 */
                var restMethod = versionOf(selfLink) === '1.0' ? AppLinks.put : AppLinks.post;
                restMethod(selfLink, data, success, error, beforeSend);
            },
            get: function(url, success, error, beforeSend) {
                return AppLinks.jsonRequest(url, 'GET', null, success, error, beforeSend);
            },
            getXml: function(url, success, error, beforeSend) {
                return AppLinks.xmlRequest(url, 'GET', null, success, error, beforeSend);
            },
            self_link: function(item) {
                for (var i = 0, _i = item.link.length; i < _i; i++) {
                    var link = item.link[i];
                    if (link.rel == "self") return link.href;
                }

                throw "No self-link found";
            },
            del: function(urlOrObject, success, error, beforeSend) {
                var url;
                if (typeof(urlOrObject) == 'string') url = urlOrObject;
                else url = AppLinks.self_link(urlOrObject);
                return AppLinks.jsonRequest(url, 'DELETE', null, success, error, beforeSend);
            },
            SPI: $.extend({}, {
                API_VERSION: "1.0",
                REST_RESOURCE_URL: AJS.contextPath() + "/rest/applinks/",
                BASE_URL: AJS.contextPath() + "/rest/applinks/1.0",
                OAUTH_REST_RESOURCE_URL: AJS.contextPath() + "/rest/applinks-oauth/",
                OAUTH_BASE_URL: AJS.contextPath() + "/rest/applinks-oauth/1.0",

                /**
                 * Update the API version and associated urls.
                 * @param version
                 */
                setApiVersion: function(version){
                    AppLinks.SPI.API_VERSION = version;
                    AppLinks.SPI.setBaseUrl(AppLinks.SPI.REST_RESOURCE_URL + AppLinks.SPI.API_VERSION);
                },
                setBaseUrl: function(url){
                    AppLinks.SPI.BASE_URL = url;
                },
                setOAuthBaseUrl: function(url){
                    AppLinks.SPI.OAUTH_BASE_URL = url;
                },
                /**
                 * Build a base URL for rest calls using the specified baseUrl.
                 * @param baseUrl
                 * @returns {string}
                 */
                getRemoteRestBaseUrl: function(baseUrl) {
                    return baseUrl + "/rest/applinks/" + AppLinks.SPI.API_VERSION;
                },
                /**
                 * Build a base URL for plugin servlet calls using the specified baseUrl.
                 * @param baseUrl
                 * @returns {string}
                 */
                getRemotePluginServletBaseUrl: function(baseUrl) {
                    return baseUrl + "/plugins/servlet";
                },
                getAllLinks: function(success, failure) {
                    var url = AppLinks.SPI.BASE_URL + "/applicationlink";
                    return AppLinks.get(url, success, failure);
                },
                getAllLinksWithAuthInfo: function(success, failure) {
                    var url = AppLinks.SPI.BASE_URL + "/listApplicationlinks";
                    return AppLinks.get(url, success, failure);
                },
                getApplicationLinkState: function(id, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + "/listApplicationlinkstates/id/" + id;
                    return AppLinks.get(url, success, failure);
                },
                getLinksOfType: function(typeId, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + "/applicationlink/type/" + typeId;
                    return AppLinks.get(url, success, failure);
                },
                tryToFetchManifest: function(url, success, failure) {
                    var restUrl = AppLinks.SPI.BASE_URL + '/applicationlinkForm/manifest.json?url=' + encodeURIComponent(url);
                    return AppLinks.get(restUrl, success, failure, function(jqxhr) {
                        jqxhr.setRequestHeader("X-Atlassian-Token", "no-check");
                    });
                },
                getManifestFor: function(id, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/manifest/' + id + ".json";
                    return AppLinks.get(url, success, failure);
                },
                getLocalManifest: function(success, failure){
                    var url = AppLinks.SPI.BASE_URL + '/manifest.json';
                    return AppLinks.get(url, success, failure);
                },
                /**
                 * Attempt to get the Manifest of the remote application, via a direct REST call.
                 * Requires CORS enabled on the REST resource.
                 * @param url
                 * @param success
                 * @param failure
                 * @returns {*}
                 */
                getRemoteManifest: function(remoteBaseUrl, success, failure){
                    var remoteManifestUrl = AppLinks.SPI.getRemoteRestBaseUrl(remoteBaseUrl) + '/manifest.json';
                    return AppLinks.get(remoteManifestUrl, success, failure);
                },
                /**
                 * Attempt to get the OAuth Consumer Info of the remote application, via a direct call.
                 * Requires CORS enabled on the REST resource.
                 * @param url
                 * @param success
                 * @param failure
                 * @returns {*}
                 */
                getRemoteOAuthConsumerInfo: function(remoteBaseUrl, success, failure){
                    var remoteManifestUrl = AppLinks.SPI.getRemotePluginServletBaseUrl(remoteBaseUrl) + '/oauth/consumer-info';
                    return AppLinks.getXml(remoteManifestUrl, success, failure);
                },
                getApplinkStatus: function (applinkId, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/status/' + applinkId;
                    return AppLinks.get(url, success, failure);
                },
                createStaticUrlAppLink: function(applicationType, success, failure) {
                    var restUrl = AppLinks.SPI.BASE_URL + '/applicationlinkForm/createStaticUrlAppLink?typeId=' + applicationType;
                    return AppLinks.post(restUrl, null, success, failure);
                },
                createLink: function(applicationLink, username, password, createTwoWayLink, customRpcUrl, rpcUrl, configFormValues, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/applicationlinkForm/createAppLink';
                    var data = {
                        applicationLink: applicationLink,
                        username: username,
                        password: password,
                        createTwoWayLink: createTwoWayLink,
                        customRpcURL: customRpcUrl,
                        rpcUrl: rpcUrl,
                        configFormValues: configFormValues
                    };
                    return AppLinks.post(url, data, success, failure);
                },
                createLinkWithOrphanedTrust : function(applicationLink, username, password, createTwoWayLink, customRpcUrl, rpcUrl, configFormValues, orphanedTrust, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/applicationlinkForm/createAppLink';
                    var data = {
                        applicationLink: applicationLink,
                        username: username,
                        password: password,
                        createTwoWayLink: createTwoWayLink,
                        customRpcURL: customRpcUrl,
                        rpcUrl: rpcUrl,
                        configFormValues: configFormValues,
                        orphanedTrust: orphanedTrust
                    };
                    return AppLinks.post(url, data, success, failure);
                },
                verifyTwoWayLinkDetails : function (remoteUrl, rpcUrl, username, password, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/applicationlinkForm/details';
                    var data = {
                        username: username,
                        password: password,
                        remoteUrl: remoteUrl,
                        rpcUrl: rpcUrl
                    };
                    return AppLinks.post(url, data, success, failure);
                },
                getApplicationLinkInfo: function (appId, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/applicationlinkInfo/id/" + appId;
                    return AppLinks.get(url, success, error);
                },
                deleteLink: function(applicationLink, reciprocate, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/applicationlink/" + applicationLink.id;
                    if (reciprocate) url += "?reciprocate=true";
                    return AppLinks.del(url, success, error);
                },
                makePrimary: function(applicationLink, success) {
                    var url = AppLinks.SPI.BASE_URL + "/applicationlink/primary/" + applicationLink.id;
                    return AppLinks.post(url, null, success);
                },
                relocate: function(applicationLink, newUrl, suppressWarnings, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/relocateApplicationlink/" + applicationLink.id + "?newUrl=" + encodeURIComponent(newUrl) +
                        "&nowarning=" + (suppressWarnings ? "true" : "false");
                    return AppLinks.post(url, null, success, error);
                },
                legacyUpgrade: function(applicationLink, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/upgrade/legacy/" + applicationLink.id;
                    return AppLinks.post(url, null, success, error);
                },
                ualUpgrade: function(applicationLink, body, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/upgrade/ual/" + applicationLink.id;
                    return AppLinks.post(url, body, success, error);
                },
                getEntityTypesForApplicationType: function(applicationType, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/type/entity/" + applicationType;
                    return AppLinks.get(url, success, error);
                },
                getLocalEntitiesWithLinksToApplication: function(applicationLinkId, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/entitylink/localEntitiesWithLinksTo/" + applicationLinkId + ".json";
                    return AppLinks.get(url, success, error);
                },
                getEntityLinksForApplication: function(applicationLinkId, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/entities/" + applicationLinkId + ".json";
                    AppLinks.get(url, success, error);
                },
                getEntityLinksForApplicationUsingAnonymousAccess: function(applicationLinkId, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/entities/anonymous/" + applicationLinkId + ".json";
                    return AppLinks.get(url, success, error);
                },
                createNonUalEntityLink: function(localType, localKey, applicationId, remoteTypeId, remoteKey, name, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/entitylink/" + localType + "/" + localKey + "?reciprocate=false";
                    var data = {
                        applicationId: applicationId,
                        typeId: remoteTypeId,
                        key: remoteKey,
                        name: name,
                        isPrimary: false
                    };
                    return AppLinks.put(url, data, success, error);
                },
                createEntityLink: function(localType, localKey, entity, reciprocate, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + "/entitylink/" + localType + "/" + localKey + "?reciprocate=";
                    url += (reciprocate ? "true" : "false");
                    return AppLinks.put(url, entity, success, failure);
                },
                getConfiguredEntityLinks: function(localType, localKey, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/entitylink/primaryLinks/" + localType + "/" + localKey + ".json";
                    return AppLinks.get(url, success, error);
                },
                deleteEntityLink: function(localTypeId, localKey, entity, reciprocate, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/entitylink/" + localTypeId + "/" + localKey + "?typeId=" + entity.typeId + "&key=" + entity.key + "&applicationId=" + entity.applicationId + "&reciprocate=" + reciprocate;
                    return AppLinks.del(url, success, error);
                },
                makePrimaryEntityLink: function(localTypeID, localKey, entity, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/entitylink/primary/" + localTypeID + "/" + localKey + "?typeId=" + entity.typeId + "&key=" + entity.key + "&applicationId=" + entity.applicationId;
                    return AppLinks.post(url, null, success, error);
                },
                canDeleteAppLink: function(applicationId, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/permission/reciprocate-application-delete/" + applicationId;
                    return AppLinks.get(url, success, error);
                },
                canDeleteEntityLink: function(localTypeId, localKey, entity, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/permission/reciprocate-entity-delete/" + entity.applicationId + "/" + localTypeId + "/" + localKey + "/" + entity.typeId + "/" + entity.key;
                    return AppLinks.get(url, success, error);
                },
                canCreateReciprocateEntityLink: function(applicationId, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/permission/reciprocate-entity-create/" + applicationId;
                    return AppLinks.get(url, success, error);
                },
                processPermissionCode: function(settings) {
                    var config = {
                        noPermission: function() {},
                        missing: function() {},
                        credentialsRequired: function(authUrl) {},
                        authenticationFailed: function(authUrl) {},
                        noAuthentication: function(authUrl) {},
                        noAuthenticationConfigured: function() {},
                        noConnection: function() {},
                        allowed: function() {},
                        unrecognisedCode: function(code) {},
                        updateView: function(message, icon, button) {}
                    };

                    if (!settings) settings = {};

                    settings = $.extend(config, settings);

                    return function(data) {
                        var code = data.code;
                        if (code == "NO_PERMISSION") {
                            settings.noPermission();
                        } else if (code == "MISSING") {
                            settings.missing();
                        } else if (code == "CREDENTIALS_REQUIRED") {
                            settings.credentialsRequired(data.url);
                        } else if (code == "AUTHENTICATION_FAILED") {
                            settings.authenticationFailed(data.url);
                        } else if (code == "NO_AUTHENTICATION") {
                            settings.noAuthentication(data.url);
                        } else if (code == "NO_AUTHENTICATION_CONFIGURED") {
                            settings.noAuthenticationConfigured();
                        } else if (code == "NO_CONNECTION") {
                            settings.noConnection();
                        } else if (code == "ALLOWED") {
                            settings.allowed();
                        } else {
                            settings.unrecognisedCode(data.code);
                        }
                    };
                },
                addAuthenticationTrigger: function(target, authUrl, callbacks) {
                    if (!callbacks) {
                        callbacks = {};
                    }

                    if (typeof callbacks.onSuccess == "undefined") {
                        callbacks.onSuccess = function() {
                            location.reload();
                        }
                    }

                    if (typeof callbacks.onFailure == "undefined") {
                        callbacks.onFailure = function() {
                            return true;
                        }
                    }
                    //Unbind previous click listener, otherwise we might end up opening multiple windows.
                    $(target).off('click.applinks');
                    $(target).on('click.applinks', function() {
                        if (callbacks.before) {
                            callbacks.before();
                        }
                        AppLinks.authenticateRemoteCredentials(authUrl, callbacks.onSuccess, callbacks.onFailure);
                    });
                },
                deleteOrphanedTrust: function(id, type, success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/orphaned-trust/" + type + "/" + id;
                    return AppLinks.del(url, success, error);
                },
                getOrphanedTrust: function(success, error) {
                    var url = AppLinks.SPI.BASE_URL + "/orphaned-trust/";
                    return AppLinks.get(url, success, error);
                },
                showCreateEntityLinkSuggestion: function() {
                    return true;
                },
                getApplicationLink: function(id, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/applicationlink/' + id;
                    return AppLinks.get(url, success, failure);
                },
                createApplicationLink: function(id, name, rpcUrl, displayUrl, typeId, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/applicationlink';
                    var data = {
                        id: id,
                        name: name,
                        rpcUrl: rpcUrl,
                        displayUrl: displayUrl,
                        typeId: typeId
                    };
                    return AppLinks.put(url, data, success, failure);
                },
// TODO APLDEV-3 extract OAuth creation code into OAuth specific js files in the Oauth plugin.
                createConsumer: function(id, key, name, description, sharedSecret, publicKey, twoLOAllowed, executingTwoLOUser, twoLOImpersonationAllowed, outgoing, success, failure) {
                    var url = AppLinks.SPI.OAUTH_BASE_URL + '/applicationlink/' + id + '/authentication/consumer';
                    var data = {
                        key: key,
                        name: name,
                        description: description,
                        sharedSecret: sharedSecret,
                        publicKey: publicKey,
                        outgoing: outgoing,
                        twoLOAllowed: twoLOAllowed,
                        executingTwoLOUser: executingTwoLOUser,
                        twoLOImpersonationAllowed: twoLOImpersonationAllowed
                    };
                    return AppLinks.put(url, data, success, failure);
                },
                createConsumerAutoConfigure: function(id, twoLOAllowed, executingTwoLOUser, twoLOImpersonationAllowed, success, failure) {
                    var url = AppLinks.SPI.OAUTH_BASE_URL + '/applicationlink/' + id + '/authentication/consumer?autoConfigure=true';
                    var data = {
                        twoLOAllowed: twoLOAllowed,
                        executingTwoLOUser: executingTwoLOUser,
                        twoLOImpersonationAllowed: twoLOImpersonationAllowed
                    };
                    return AppLinks.put(url, data, success, failure);
                },
                registerProvider: function(id, provider, config, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/applicationlink/' + id + '/authentication/provider';
                    var data = {
                        config : config,
                        provider : provider
                    };
                    return AppLinks.put(url, data, success, failure);
                },
                enableFeature: function(featureName, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/features/' + featureName;
                    return AppLinks.put(url, {}, success, failure);
                },
                disableFeature: function(featureName, success, failure) {
                    var url = AppLinks.SPI.BASE_URL + '/features/' + featureName;
                    return AppLinks.del(url, success, failure);
                }
            }, (window.AppLinks && window.AppLinks.SPI) || {})
        });

        AppLinks.UI = {
            showInfoBox: function(message) {
                $('.aui-message.aui-message-success').remove();
                AppLinks.UI.createMessage('success', message, 'page-info');
            },
            hideInfoBox: function() {
                $('.aui-message.aui-message-success').remove();
            },
            showErrorBox: function(message) {
                AppLinks.UI.createMessage('error', message, 'page-error');
            },
            hideErrorBox: function() {
                $('.aui-message.aui-message-error').remove();
            },
            showWarningBox: function(messages) {
                if ($.isArray(messages) && messages.length > 0) {
                    var ulEl = $("<ul></ul>");
                    $(messages).each(function(index) {
                        ulEl.append($("<li/>", {
                            text: messages[index]
                        }));
                    });
                    var messageEl = $('<div class="page-warning"></div>').append(ulEl);
                    AppLinks.UI.createMessage('warning', messageEl.html(), 'page-warning');
                } else {
                    AppLinks.UI.createMessage('warning', messages, 'page-warning');
                }
            },
            hideWarningBox: function() {
                $('.aui-message.aui-message-warning').remove();
            },
            shortenString: function(message, maxLength) {
                if (message.length  > maxLength) {
                    message = message.substring(0, maxLength) + "...";
                }
                return message;
            },
            createMessage: function(type, message, cssClass) {
                var messageEl = $('<div class="' + cssClass + '">');
                messageEl.html(message);
                AJS.messages[type](".applinks-message-bar", {
                    title: "",
                    body: messageEl.wrap('<div></div>').parent().html(),
                    closeable: true
                });
                $(document).trigger(AppLinks.Event.Legacy.MESSAGE_BOX_DISPLAYED);
            },
            displayValidationErrorMessages: function (errorClass, rootEl, messages) {
                if ($.isArray(messages)) {
                    $(messages).each(function(i,v) {
                        var d = $('<div class="error applinks-error">');
                        d.text(v);
                        $(rootEl).find("." + errorClass).append(d);
                    });
                } else if(typeof messages != 'undefined'){
                    var d = $('<div class="error applinks-error">');
                    d.text(messages.toString());
                    $(rootEl).find("." + errorClass).append(d);
                }
            },
            displayValidationError: function(errorClass, rootEl, errorFn) {
                return function(xhr) {
                    if (xhr.status == 401) {
                        window.location.reload();
                        return;
                    }
                    $('.applinks-error').remove();
                    $('aui-spinner').remove();

                    var respObj = parseJsonResponse(xhr);
                    var messages = respObj.message;

                    if (typeof respObj.fields == "undefined") {
                        AppLinks.UI.displayValidationErrorMessages(errorClass, rootEl, messages);
                    } else {
                        var fields = respObj.fields;
                        $(fields).each(function(index) {
                            var d = $('<div class="error applinks-error" id="' + fields[index] + '-error">');
                            d.text(messages[index]);
                            if ($(rootEl).find('.' + fields[index]).length > 0) {
                                d.insertAfter($(rootEl).find('.' + fields[index]));
                            } else {
                                d.insertAfter($(rootEl).find('.' + errorClass).append(d));
                            }
                        });
                    }
                    $(rootEl).find('.' + errorClass).addClass("fully-populated-errors");
                    if (errorFn) {
                        errorFn();
                    }
                }
            },
            addProtocolToURL : function(url) {
                var newUrl = $.trim(url);
                var tempURL = newUrl.toLowerCase();
                var hasProtocol = false;
                if (tempURL.length >= 7) {
                    if (tempURL.substring(0,7).indexOf('http') != -1) {
                        hasProtocol = true;
                    }
                }
                //default protocol is http
                if (!hasProtocol) {
                    newUrl = 'http://' + newUrl;
                }
                return newUrl;
            },
            /**
             * Similar to the standard Javascript join() method, but nicer in that
             * it uses a different delimiter for the last node (by default "and"),
             * so that:
             * {code}
             * "1, 2 and 3" == prettyJoin(['1', '2', '3'], function(value) {return value;});
             * {code}
             *
             * @param inputArray
             * @param resolveFn
             * @param finalDelimiter
             */
            prettyJoin : function(inputArray, resolveFn, finalDelimiter) {
                if (!finalDelimiter) {
                    finalDelimiter = "and";
                }
                var maxLength = inputArray.length;
                var message = "";
                $.each(inputArray, function(index, value) {
                    if (index == (maxLength - 1) && maxLength > 1) {
                        message += " " + finalDelimiter + "  " + resolveFn(value);
                    } else {
                        message += resolveFn(value);
                        if (index + 2 < maxLength) {
                            message += ", ";
                        }
                    }
                });
                return message;
            },
            showLoadingIcon: function(element) {
                $('<aui-spinner size="small" class="applinks-loading"></aui-spinner>').insertBefore(element);
            },
            hideLoadingIcon: function(element) {
                $(element).prev('aui-spinner').remove();
            },
            findUrl: function(text) {
                var url = undefined;
                var lcText = text.toLowerCase();
                var startOfUrl = lcText.indexOf('http:');
                if (startOfUrl == -1) {
                    startOfUrl = lcText.indexOf('https:');
                }
                if (startOfUrl > -1) {
                    var endOfUrl = lcText.indexOf(' ', startOfUrl);
                    if (endOfUrl == -1) {
                        endOfUrl = lcText.length;
                    }
                    url = text.substring(startOfUrl, endOfUrl); // use _case-sensitive_ version to retrieve the actual URL
                }
                return url;
            },
            findApplicationType : function(id) {
                id = id.toLowerCase();
                if (id.indexOf("jira") != -1) {
                    return "jira";
                } else if (id.indexOf("fisheye") != -1) {
                    return "fecru";
                } else if (id.indexOf("confluence") != -1) {
                    return "confluence";
                } else if (id.indexOf("refapp") != -1) {
                    return "refapp";
                } else {
                    return undefined;
                }
            },
            escapeSelector: function(selector) {
                // based on http://samuelsjoberg.com/archive/2009/09/escape-jquery-selectors
                return selector.replace(/([#;&,\.\+\*\~':"\!\^$\[\]\(\)=>\|])/g, "\\$1");
            },
            sanitiseHTML: function(input) {
                var replacements = {
                    "<": "&lt;",
                    '"': "&quot;",
                    "&": "&amp;"
                };
                return input.replace(/[<"&]/g, function(match) {
                    return replacements[match];
                });
            },
            refreshOrphanedTrust: function() {
                // post dialog -- check whether we need to remove any orphaned-trust entries
                var updateOrphanedTrust = function(data) {
                    $("tr.orphaned-trust-row").each(function() {
                        var $this = $(this);
                        var id = $this.attr("data-id");
                        var type = $this.attr("data-type");
                        var stillExists = false;
                        for (var i = 0; i < data.orphanedTrust.length; i++) {
                            var ot = data.orphanedTrust[i];
                            if (id == ot.id && type == ot.type) {
                                stillExists = true;
                                break;
                            }
                        }
                        if (!stillExists) {
                            $this.remove();
                            if (data.orphanedTrust.length == 0) {
                                // we just removed the last orphaned trust cert, hide warning!
                                $(".orphaned-trust-warning").hide();
                            }
                        }
                    });
                };

                AppLinks.SPI.getOrphanedTrust(updateOrphanedTrust);
            },
            removeCssClass: function(element, prefix) {
                $(element).removeClass( function(index, className) {
                    var classes = className.split(' ');
                    var classToRemove = "";
                    $.each(classes, function(index, value) {
                        if (value.indexOf(prefix) != -1) {
                            classToRemove = value;
                        }
                    });
                    return classToRemove;
                } );
            }
        };

        /**
         * Add jQuery event system to AppLinks.UI namespace.
         */
        (function(){
            var eventBus = $({});
            $.each(['bind', 'unbind', 'trigger'], function(i, current){
                AppLinks.UI[current] = function(){
                    return eventBus[current].apply(eventBus, arguments);
                }
            });
        })();

        $(document).trigger(AppLinks.Event.PREREADY);
        $(document).trigger(AppLinks.Event.READY);
    });
})(require('applinks/common/i18n'), require('applinks/common/docs'), require('applinks/common/events'));

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-util-js', location = 'applinks/internal/non-amd/autocomplete.js' */
// NOTE: this is used outside of Applinks. See atlassian-plugin.xml for more details about the associated restrictions

AJS.$(document).bind(AppLinks.Event.READY, function() {
/**
 * TODO: THIS CODE IS COPIED FROM CONFLUENCE and should be part of AUI/AJS.
 * IF AUI comes with the InputDrivenDropDown when can remove this file.
 * https://studio.atlassian.com/browse/AJS-471
 *
 * A simple cache manager that supports a
 * FIFO cache invalidation strategy.
 *
 * @class cacheManager
 * @namespace AJS.Confluence
 * @constructor
 * @param cacheSize the size of the cache before keys are invalidated
 */
AppLinks.autoComplete = {
        cacheManager : function (cacheSize) {
    var cache = {},
        cacheStack = [],
        cacheSize = cacheSize || 30;

    return {
        /**
         * Return the value stored in the cache for the given key
         * @method get
         * @param key {String}
         */
        get: function(key) {
            return cache[key];
        },
        /**
         * Put the given key, value in the cache
         * @method put
         * @param key {String}
         * @param value {Object}
         */
        put: function(key, value) {
            cache[key] = value;
            cacheStack.push(key);
            if (cacheStack.length > cacheSize) {
                delete cache[cacheStack.shift()];
            }
        },
        /**
         * Clear the cache.
         */
        clear : function() {
            cache = {};
            cacheStack = [];
        }
    };
}};


(function($){
    /**
     * Check that all items in the drop down can be displayed - show ellipses at the end of any that
     * are too long. Also remove any unused properties that the dropDown may have stored for each
     * item in the list.
     *
     * @method truncateText
     * @private
     */
    var truncateText = function (dd) {
        AJS.log("InputDrivenDropDown: truncating text");
        var width = dd.$.closest(".aui-dropdown").width(),
            rightPadding = 20; // add some padding so the ellipsis doesn't run over the edge of the box

        $("a span:not(.icon)", dd.$).each(function () {
            var $a = $(this),
                elpss = $("<var></var>").html("&#8230;"),
                elwidth = elpss.width(),
                isLong = false;

            $a.wrapInner($("<em>"));
            $("em", $a).each(function () {
                var $label = $(this);

                $label.show();
                if (this.offsetLeft + this.offsetWidth > width) {
                    var childNodes = this.childNodes,
                        success = false;

                    for (var j = childNodes.length - 1; j >= 0; j--) {
                        var childNode = childNodes[j],
                            truncatedChars = 1,
                            valueAttr = (childNode.nodeType == 3) ? "nodeValue" : "innerHTML",
                            nodeText = childNode[valueAttr];

                        do {
                            if (truncatedChars <= nodeText.length) {
                                childNode[valueAttr] = nodeText.substr(0, nodeText.length - truncatedChars++);
                            } else { // if we cannot fit even one character of the next word, then try truncating the node just previous to this
                                break;
                            }
                        } while (this.offsetLeft + this.offsetWidth + elwidth > width - rightPadding);

                        if (truncatedChars <= nodeText.length) {
                            // we've managed truncate part of the word and fit it in
                            success = true;
                            break;
                        }
                    }

                    if (success) {
                        isLong = true;
                    } else {
                        $label.hide();
                    }
                }
            });
            if (isLong) {
                $a.append(elpss);
                this.elpss = elpss;
            }
        });
    };

    var highlightTokens = function(dd, tokens) {
        if (!tokens.length || !tokens[0]) return;

        AJS.log("InputDrivenDropDown: highlighting tokens");

        // escape regex chars .*+?|()[]{}\ first
        for (var i = 0, ii = tokens.length; i < ii; i++) {
            var token = tokens[i];
            tokens[i] = token ? token.replace(/[\.\*\+\?\|\(\)\[\]{}\\]/g, "\\$") : "";
        }

        var regex = new RegExp("(" + tokens.join("|") + ")", "gi");

        $("li a:not(.dropdown-prevent-highlight) span", dd.$).each(function() {
            var span = $(this),
                html = span.html().replace(regex, "<strong>$1</strong>");
            span.html(html);
        });
    };

    /**
     * Builds and shows the dropdown.
     *
     * @param idd the InputDrivenDropdown
     * @param dropdownData in the form { matrix, query, queryTokens }
     * @private
     */
    var makeDropdown = function (idd, dropdownData) {
        var options = idd.options,
            old_dd = idd.dd;

        if (old_dd) {
            old_dd.hide();
            old_dd.$.remove();
        }

        options.ajsDropDownOptions = options.ajsDropDownOptions || {};
        if (options.ajsDropDownOptions && !options.ajsDropDownOptions.alignment) { // default to left alignment
            options.ajsDropDownOptions.alignment = "left";
        }
        //this needs to be moved into aui
        options.ajsDropDownOptions.selectionHandler = options.ajsDropDownOptions.selectionHandler || function(e, element) {
            if(e.type != "click") {
                e.preventDefault();
                $("a",element).click();
                document.location = $("a",element).attr("href");
            }
        };

        /* Fixing an AUI bug in here:  AJS.dropdown puts the raw 'matrix[i].name' as html, without escaping it !
           The solution is to override their displayHandler
         */
        /**
         * Escape obj.name and return it
         */
        options.ajsDropDownOptions.displayHandler = function (obj) {
            return AJS.escapeHtml(obj.name);
        }

        var dd = idd.dd = new AJS.dropDown(dropdownData.matrix, options.ajsDropDownOptions)[0];

        // could move into dropdown.js in AUI
        if (options.ajsDropDownOptions && options.ajsDropDownOptions.className) {
            dd.$.addClass(options.ajsDropDownOptions.className);
        }

        // place the created drop down using the configured dropdownPlacement function
        // if there is none then use a default behaviour
        if (options.dropdownPlacement) {
            options.dropdownPlacement(dd.$);
        } else {
            AJS.log("No dropdownPlacement function specified. Appending dropdown to the body.");
            $("body").append(dd.$);
        }

        highlightTokens(dd, dropdownData.queryTokens || [dropdownData.query]);
        truncateText(dd);

        if (options.dropdownPostprocess) {
            options.dropdownPostprocess(dd.$);
        }
        dd.show(idd._effect);

        if (typeof options.onShow == "function") {
            options.onShow.call(dd, dd.$);
        }

        return dd;
    };

    /**
     * Provides a controller-agnostic object that listens for controller changes and populates a dropdown
     * via a callback. Most aspects can be customized via the options object parameter.
     * <br>
     * Options are:
     * <li>
     *   getDataAndRunCallback - (required) callback method used to provide data for the dropdown. It must take
     *                          two parameters, user input value and the callback function to execute.
     * </li>
     * <li>
     *   onShow - function to call when the drop-down is displayed
     * </li>
     * <li>
     *   dropdownPlacement - a function that will be called with the drop down and which should place it in the
     *                          correct place on the page. The supplied arguments are 1) the input that issued the
     *                          search, 2) the dropDown to be placed.
     * </li>
     * <li>
     *   ajsDropDownOptions - any options the underlying dropDown component can handle expects
     * </li>
     * <li>
     *   onDeath - callback to run when dropdown dies
     * </li>
     * @class InputDrivenDropDown
     * @namespace AJS
     */
    function InputDrivenDropDown(id, options) {
        this._effect = "appear";
        this._timer = null;

        this.id = id;
        this.options = options;
        this.inactive = false;
        this.busy = false;
        this.cacheManager = AppLinks.autoComplete.cacheManager();
    }

    /**
     * Clears the cache.
     */
    InputDrivenDropDown.prototype.clearCache = function () {
        this.cacheManager.clear();
    };

    /**
     * This method should be called when the user input for this dropdown has changed.
     * It will check the cache before fetching data (via options.getDataAndRunCallback)
     * and displaying the dropdown.
     *
     * @param value {String} the new value of the user input
     * @param force {Boolean} force a change to occur regardless of user input
     */
    InputDrivenDropDown.prototype.change = function (value, force) {
        var t = this;
        if (value != t._value || force) {
            t._value = value;
            t.busy = false;

            clearTimeout(t._timer);

            if (force || (/\S{0,}/).test(value)) {
                var cachedVal = t.cacheManager.get(value);
                if (cachedVal) {
                    makeDropdown(t, cachedVal);
                } else {
                    t.busy = true;
                    t._timer = setTimeout(function () { // delay sending a request to give the user a chance to finish typing their search term(s)
                        t.options.getDataAndRunCallback.call(t, value, t.show);
                    }, 200);
                }
            } else {
                t.dd && t.dd.hide();
            }
        }
    };

    /**
     * Gets the number of visible options in the dropdown.
     */
    InputDrivenDropDown.prototype.dropDownLength = function () {
        return this.dd.links ? this.dd.links.length : 0;
    };
    
    /**
     * Gets the specified menu item from the dropdown list.
     * 
     * @param index {Integer} the 0-based index of the dropdown option list
     */
    InputDrivenDropDown.prototype.dropDownItem = function (index) {
        return this.dropDownLength() > index ? this.dd.links[index] : null;
    };
    
    /**
     * Hides the drop down
     */
    InputDrivenDropDown.prototype.hide = function () {
        this.dd && this.dd.hide();
    };

    /**
     * Hides and removes the drop down from the DOM.
     */
    InputDrivenDropDown.prototype.remove = function () {
        var dd = this.dd;
        if (dd) {
            this.hide();
            dd.$.remove();
        }
        this.inactive = true;
        this.options.onDeath && this.options.onDeath();
    };

    /**
     * Shows the drop down with the given matrix data and query.
     * <br>
     * Matrix property should be an array of arrays, where the sub-arrays represent the different
     * search categories.
     *
     * Expected properties of category sub-array objects are:
     *  - href
     *  - name
     *  - className
     *  - html (optional, replaces href and name)
     *  - icon (optional)
     *
     *
     * @param matrix {Array} matrix to populate the drop down from
     * @param query {String} the user input string that triggered this show
     * @param queryTokens {Array} an array of strings of the query tokens. Use for highlighting search terms.
     */
    InputDrivenDropDown.prototype.show = function (matrix, query, queryTokens) {
        if (this.inactive) {
            AJS.log("Quick search abandoned before server response received, ignoring. " + this);
            return;
        }

        var dropdownData = {
            matrix: matrix,
            query: query,
            queryTokens: queryTokens
        };
        this.cacheManager.put(query, dropdownData);

        makeDropdown(this, dropdownData);
        this.busy = false;
    };

    /**
     * Returns an InputDrivenDropDown. See InputDrivenDropDown for more documentation.
     * @param options {Object} options for the InputDrivenDropDown
     * @constructor
     */
    AppLinks.inputDrivenDropdown = function (options) {
        return new InputDrivenDropDown("inputdriven-dropdown", options);
    };

})(jQuery);
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-util-js', location = 'js/fecru-compatibility.js' */
if (jQuery != undefined && AJS != undefined) jQuery = AJS.$; // make sure we're extending the correct jQuery
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:feature-oauth-dance', location = 'applinks/internal/feature/oauth/oauth-callback.js' */
define('applinks/feature/oauth-callback', [
    'applinks/lib/window',
    'applinks/lib/lodash',
    'applinks/common/preconditions'
], function(
    window,
    _,
    Preconditions
) {
    function OAuthCallback(url) {
        Preconditions.nonEmptyString(url, 'url');
        this._url = url;
    }

    OAuthCallback.prototype.source = function(source) {
        Preconditions.hasValue(source, 'source');
        this._source = source;
        return this;
    };

    OAuthCallback.prototype.onSuccess = function(callback) {
        Preconditions.isFunction(callback, 'onSuccess');
        this._onSuccess = callback;
        return this;
    };

    OAuthCallback.prototype.onFailure = function(callback) {
        Preconditions.isFunction(callback, 'onFailure');
        this._onFailure = callback;
        return this;
    };

    // API for OAuth to invoke callbacks
    OAuthCallback.prototype.success = function() {
        this.oauthWindow.close();
        if (this._onSuccess) {
            this._onSuccess(this._source);
        }
        // free up the oauthCallback object
        delete window.oauthCallback;
    };

    OAuthCallback.prototype.failure = function() {
        this.oauthWindow.close();
        if (this._onFailure) {
            this._onFailure(this._source);
        }
        // free up the oauthCallback object
        delete window.oauthCallback;
    };

    // start the dance
    OAuthCallback.prototype.open = function() {
        // required for OAuth callbacks to fire
        window.oauthCallback = this;
        this.oauthWindow = window.open(this._url, 'com_atlassian_applinks_authentication');
    };

    return OAuthCallback;
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:feature-oauth-dance', location = 'applinks/internal/feature/oauth/oauth-dance.js' */
define('applinks/feature/oauth-dance', [
    'applinks/lib/console',
    'applinks/lib/jquery',
    'applinks/lib/lodash',
    'applinks/lib/window',
    'applinks/common/events',
    'applinks/common/preconditions',
    'applinks/feature/oauth-callback'
], function(
    console,
    $,
    _,
    window,
    ApplinksEvents,
    Preconditions,
    OAuthCallback
) {
    /**
     * Creates a new OAuth Dance. This initializes elements found by `selector` within `scope` to initiate the OAuth
     * dance on click. The elements need to have a `data-authorisation-uri` attribute that points to the authorisation
     * page to open.
     *
     * @param scope {string} optional selector for scope, if not defined then global `document` will be used
     * @param selector {string} selector for the DOM elements to initialize OAuth dance for. If not defined then scope will be used
     * @constructor
     */
    function OAuthDance(scope, selector) {
        this._scope = scope || window.document;
        this._selector = selector;
    }

    OAuthDance.prototype.onSuccess = function(callback) {
        Preconditions.isFunction(callback, 'onSuccess');
        this._onSuccess = callback;
        return this;
    };

    OAuthDance.prototype.onFailure = function(callback) {
        Preconditions.isFunction(callback, 'onFailure');
        this._onFailure = callback;
        return this;
    };

    OAuthDance.prototype.defaultSuccess = function() {
        return this.onSuccess(function() {
            window.location.reload()
        });
    };

    OAuthDance.prototype.defaultFailure = function() {
        return this.onFailure(function() { return true });
    };

    /**
     * Wire up click events for the selected elements to initiate OAuth dance.
     */
    OAuthDance.prototype.initialize = function() {
        var that = this;
        if (this._selector) {
            $(this._scope).on('click', this._selector, function(e) {
                e.preventDefault();
                that._open($(this));
            });
        } else {
            $(this._scope).on('click', function(e) {
                e.preventDefault();
                that._open($(this));
            });
        }
    };

    /**
     * Start the OAuth dance for the given selector/scope.
     */
    OAuthDance.prototype.start = function() {
        var $scope = $(this._scope);
        var $element = this._selector ? $scope.find(this._selector) : $scope;
        this._open($element);
    };

    OAuthDance.prototype._open = function(element) {
        if (element.length !== 1) {
            console.warn('Could not trigger OAuth dance, the source is not a single HTML element: ' + element);
            return;
        }
        
        var authorisationUri = element.attr('data-authorisation-uri');

        if (authorisationUri) {
            this._onSuccess || this.defaultSuccess();
            this._onFailure || this.defaultFailure();
            new OAuthCallback(authorisationUri)
                .source(element)
                .onSuccess(this._onSuccess)
                .onFailure(this._onFailure)
                .open();
        } else {
            console.warn('Could not trigger OAuth dance, data-authorisation-uri missing for: ' + element);
        }
    };

    return OAuthDance;
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.applinks.applinks-plugin:applinks-oauth-ui', location = 'js/oauth-dialog.js' */
// NOTE: this is used outside of Applinks. See atlassian-plugin.xml for more details about the associated restrictions

/**
 * @deprecated use applinks/feature/oauth-callback and applinks/feature/oauth-dance AMD module
 */
(function($, ApplinksEvents, OAuthCallback) {
    // rest-service.js will load the AppLinks name space
    AppLinks.authenticateRemoteCredentials = function (url, onSuccess, onFailure) {
        $('.applinks-error').remove();
        new OAuthCallback(url).onSuccess(onSuccess).onFailure(onFailure).open();
    };
})(require('applinks/lib/jquery'), require('applinks/common/events'), require('applinks/feature/oauth-callback'));

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.extra.jira:common', location = 'util/retry-caller.js' */
define("confluence/jim/util/retry-caller",["ajs","jquery","underscore"],function(r,d,h){return function(k,a){a=a||{};var l=a.args,m="function"===typeof a.tester?a.tester:function(){return!0},e=a.delays||[.1,.3,.5,.7,1],n=e.length,c=d.Deferred(),p=c.promise(),b=a.context||c,f=0,g=function(){if(f===n)return c.rejectWith(b,[b,"exceed-maximum-called-times",""]);var q=e[f++];h.delay(function(){d.when(k.apply(b,l)).then(function(){c.notify.apply(b,arguments);m.apply(b,arguments)?g():c.resolveWith(b,arguments)},
function(){c.rejectWith(b,arguments)})},q)};g();return p}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.extra.jira:common', location = 'util/common.js' */
AJS.JiraIssues={Remote:{}};var appLinksI18n={entries:{}};
jQuery(document).ready(function(){AJS.JiraIssues=jQuery.extend(AJS.JiraIssues||{},{bindOAuthLink:function(a,b){var d=function(){b()},e=function(){},k=a.attr("href");a.click(function(l){AppLinks.authenticateRemoteCredentials(k,d,e);l.preventDefault()})},getOAuthRealm:function(a){a=a.getResponseHeader("WWW-Authenticate")||"";return(a=/OAuth realm="([^"]+)"/.exec(a))?a[1]:null}});jQuery("a.static-oauth-init").each(function(){AJS.JiraIssues.bindOAuthLink(jQuery(this),function(){window.location.reload()})});
jQuery("a.anonymous-init").each(function(a,b){a=encodeURIComponent(window.location.pathname.replace(Confluence.getContextPath(),""));a=Confluence.getContextPath()+"/login.action?os_destination\x3d"+a;AJS.$(b).attr("href",a)});var w=function(a){for(var b=AJS.JiraIssues.Remote[a],d="",e=0;e<b.length;e++)d+=b[e].key+(e<b.length-1?",":"");var k=function(c){c="/sr/jira.issueviews:searchrequest-xml/temp/SearchRequest.xml?jqlQuery\x3d"+encodeURIComponent("issuekey in ("+c+")")+"\x26returnMax\x3dtrue";return contextPath+
"/plugins/servlet/issue-retriever?appId\x3d"+a+"\x26url\x3d"+encodeURIComponent(c)+"\x26columns\x3dsummary\x26columns\x3dtype\x26columns\x3dresolution\x26columns\x3dstatus"},l=function(c){AJS.$("item",c).each(function(){for(var m=AJS.$("link",this).text(),h=AJS.$("key",this).text(),f=AJS.$("summary",this).text(),p=AJS.$("type",this),g="-1"!=AJS.$("resolution",this).attr("id"),v=AJS.$("status",this),r=AJS.$(".unknown-jira-issue."+h),q=0;q<r.length;q++){var t=AJS.$("\x3ca style\x3d\"background-image: url('"+
p.attr("iconUrl")+'\')" href\x3d"'+m+'"\x3e\x3c/a\x3e');t.text(h);var u=AJS.$('\x3cspan class\x3d"jira-status"\x3e\x3c/span\x3e');u.text(v.text().toUpperCase());var n=AJS.$('\x3cspan class\x3d"jira-issue'+(g?" resolved":"")+'" \x3e\x3c/span\x3e');n.append(t);n.append(document.createTextNode(" - "+f+" - "));n.append(u);AJS.$(r[q]).replaceWith(n)}})};d=k(d);AJS.$.ajax({url:d,success:l,error:function(c){if(401==c.status){var m=AJS.JiraIssues.getOAuthRealm(c);if(m){var h={};AJS.$(b).each(function(){if(!h[this.key]){h[this.key]=
!0;var f=AJS.$('\x3cspan class\x3d"oauth-msg"\x3e - \x3ca class\x3d"oauth-init" href\x3d"'+m+'"\x3e'+"Authenticate"+"\x3c/a\x3e "+"to see issue details"+"\x3c/span\x3e");AJS.$(".unknown-jira-issue."+this.key).addClass("single-issue-oauth").append(f)}});AJS.JiraIssues.bindOAuthLink(AJS.$(".single-issue-oauth a.oauth-init"),function(){window.location.reload()})}}else 400==c.status&&1<b.length&&AJS.$(b).each(function(){var f=this.key,p=k(f);
AJS.$.ajax({url:p,success:l,error:function(g){g=AJS.$(".unknown-jira-issue."+f);g.removeClass("single-issue-oauth");AJS.$(".oauth-msg",g).remove();g.addClass("jira-error")}})})}})};AJS.$(".unknown-jira-issue").each(function(a,b){b=AJS.$(b);a=b.attr("data-app-link");b=b.attr("data-key");AJS.JiraIssues.Remote[a]=AJS.JiraIssues.Remote[a]||[];AJS.JiraIssues.Remote[a].push({key:b})});for(var x in AJS.JiraIssues.Remote)w(x)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.extra.jira:web-resources', location = 'templates/extra/jira/jiraIssues.js' */
jQuery(document).ready(function(){var k=jQuery.extend(window.JiraIssues||{},{ADG_ENABLED:4E3<=AJS.Meta.getNumber("build-number"),ADG_FONT_SIZE_OVER_FLEXIGRID_FONT_SIZE_RATIO:14/11,onSuccessFunction:function(b){if(!jQuery("fieldset input[name\x3d'height']",b).length){var a=jQuery(".bDiv table[id^\x3d'jiraissues_table']",b)[0],c=a.grid;a=a.clientHeight+jQuery(".hDiv",b)[0].clientHeight;jQuery(".bDiv",b).css("height",a+"px");c.options.height=a;c.fixHeight(a)}},onErrorFunction:function(b,a,c,e,d){c=jQuery("#"+
a);d=e.getResponseHeader("WWW-Authenticate")||"";if("401"==e.status&&-1!=d.indexOf("OAuth")){if(e=/OAuth realm="([^"]+)"/.exec(d)){c.empty();k.bigMessageFunction(a,'\x3ca class\x3d"oauth-init" href\x3d"'+e[1]+'"\x3e'+"Authenticate"+"\x3c/a\x3e "+"to retrieve your issues"+"\x3c/span\x3e");jQuery(".bmDiv",b).css({"z-index":2});var l=function(){window.location.reload()},h=function(){};a=jQuery(".oauth-init",c.parent());var m=a.attr("href");a.click(function(g){AppLinks.authenticateRemoteCredentials(m,
l,h);g.preventDefault()});AJS.$(".gBlock").remove()}}else if("400"==e.status)k.bigMessageFunction(a,"The Jira server was not able to process the search. This may indicate a problem with the syntax of this macro. Alternatively, if this table is requesting specific issue keys, you may not have permission to view one of these issues.");else{var f=jQuery("iframe.jiraissues_errorMsgSandbox",b);f.load(function(){var g=this.contentWindow||this.contentDocument;g=jQuery((g.document?g.document:g).body);jQuery("a",g).each(function(){this.target="_top"});jQuery(".pPageStat",b).empty().text(g.text());g=jQuery("div.bmDiv",b)[0];f.removeClass("hidden");f.css({height:g.clientHeight+"px",
width:g.clientWidth+"px"})});f[0].src=jQuery("fieldset input[name\x3d'retrieverUrlHtml']",b).val();k.bigMessageFunction(a,f)}jQuery(b).find(".pReload").removeClass("loading");c[0].grid.loading=!1;jQuery(b).find(".pButton").each(function(){jQuery(this).removeClass("pBtnOver");jQuery(this).css({cursor:"default",opacity:"0.3"})});jQuery(b).find("span.pcontrol input").attr("readonly","true")},onReloadFunction:function(b,a,c){jQuery(".bmDiv",a).remove();jQuery(".bmDistance",a).remove();c.onSubmit=function(){k.reloadOnSubmitFunction(b,
c);return!0}},reloadOnSubmitFunction:function(b,a){a.params=[{name:"useCache",value:!1}];a.onSubmit=function(){k.onSubmitFunction(b,a);return!0}},onSubmitFunction:function(b,a){a.params=[{name:"useCache",value:b}]},showTrustWarningsFunction:function(b,a){b=jQuery(b).children(".trusted_warnings");a.trustedMessage?(b.find("td:last").html(a.trustedMessage),b.css("display","block")):b.css("display","none")},preProcessFunction:function(b,a,c,e,d){c&&k.showTrustWarningsFunction(b,e);0==e.total&&(jQuery(".pPageStat",
b).html(d),k.bigMessageFunction(a,d),jQuery(".pReload",b).removeClass("loading"))},bigMessageFunction:function(b,a){var c=jQuery("\x3cdiv\x3e\x3c/div\x3e"),e=jQuery("\x3cdiv\x3e\x3c/div\x3e");c.addClass("bmDistance");e.addClass("bmDiv");"string"==typeof a?e.html("\x3cp\x3e\x3cstrong\x3e"+a+"\x3c/strong\x3e\x3c/p\x3e"):a.appendTo(e);jQuery("#"+b).after(e).after(c)},getParamsFrom:function(b){var a={};b.children("input").each(function(){a[jQuery(this).attr("name")]=jQuery(this).attr("value")});return a},
initializeColumnWidth:function(b,a){var c={},e=function(p){return k.ADG_ENABLED?Math.round(p*k.ADG_FONT_SIZE_OVER_FLEXIGRID_FONT_SIZE_RATIO):p};if(!a||!a.length)return c;b=b.width()-(37+11*a.length);for(var d=!1,l=!1,h=0,m=e(140),f=0,g=a.length;f<g;f++){var n=a[f].name;switch(n){case "summary":d=!0;h++;break;case "description":l=!0;h++;break;case "type":h++;c[n]=30;b-=30;break;case "priority":h++;c[n]=50;b-=50;break;case "status":h++;c[n]=e(100);b-=e(100);break;case "key":h++;c[n]=e(90);b-=e(90);
break;case "comments":case "attachments":case "version":case "component":case "resolution":h++;c[n]=e(80);b-=e(80);break;default:c[n]=m}}if(d||l)b-=m*(a.length-h),d&&l?(c.summary=Math.max(b/2,250),c.description=Math.max(b/2,250)):d?c.summary=Math.max(b,250):c.description=Math.max(b,250);else if(!d&&!l&&a.length>h)for(m=b/(a.length-h),f=0;f<g;f++)({resolution:!0,key:!0,type:!0,priority:!0,status:!0})[a[f]]||(c[a[f]]=m);return c}});jQuery(".jiraissues_table").each(function(b,a){var c=jQuery(a),e=c.children("fieldset"),
d=k.getParamsFrom(e),l="jiraissues_table_"+b;c.append('\x3ctable id\x3d"'+l+'" style\x3d"display:none"\x3e\x3c/table\x3e');c.css("width",d.width);var h=[];e.children(".columns").each(function(f){var g=jQuery(this).hasClass("nowrap");h[f]={display:this.name,name:this.value,nowrap:g,sortable:!0,align:"left"}});var m=k.initializeColumnWidth(c,h);jQuery.each(h,function(f,g){g.width=m[g.name]});b=jQuery("\x3cdiv\x3e\x3c/div\x3e");jQuery("\x3ca\x3e\x3c/a\x3e").attr({rel:"nofollow",href:d.clickableUrl}).text(d.title).appendTo(b);
jQuery("#"+l).flexigrid({url:d.retrieverUrlHtml,method:"GET",dataType:"json",colModel:h,sortname:d.sortField,sortorder:d.sortOrder,usepager:!0,title:b.html(),page:parseInt(d.requestedPage,10),useRp:!1,rp:parseInt(d.resultsPerPage,10),showTableToggleBtn:!0,height:d.height?parseInt(d.height,10):480,onSuccess:function(){k.onSuccessFunction(a)},onSubmit:function(){k.onSubmitFunction(d.useCache,this);return!0},preProcess:function(f){k.preProcessFunction(a,l,d.showTrustWarnings,f,d.nomsg);return f},onError:function(f,
g,n){k.onErrorFunction(a,l,d.jiraissuesError,f,g,n)},onReload:function(){k.onReloadFunction(d.useCache,a,this);return!0},errormsg:d.errormsg,pagestat:d.pagestat,procmsg:d.procmsg,nomsg:d.nomsg})});jQuery(".jiraissues_count").each(function(b,a){var c=jQuery(a);jQuery.ajax({cache:!1,type:"GET",url:c.find(".url").text(),data:{useCache:c.find(".use-cache").text(),rp:c.find(".rp").text(),showCount:"true"},success:function(e){var d=c.find(".result");1==e?d.text(AJS.format("{0} issue",
e)):d.text(AJS.format("{0} issues",e));d.removeClass("hidden");jQuery(".calculating, .error, .data",c).remove()},error:function(e){var d=jQuery(".error",c).removeClass("hidden"),l=e.getResponseHeader("WWW-Authenticate")||"",h=!1;if(401===e.status&&-1!=l.indexOf("OAuth")){var m=/OAuth realm="([^"]+)"/.exec(l);m&&(d.empty().append(AJS.$("\x3ca/\x3e",{href:m[1],"class":"oauth-init"}).text("Authenticate").click(function(){AppLinks.authenticateRemoteCredentials(m[1],
function(){window.location.reload()},function(){});return!1})).append(AJS.$("\x3cspan/\x3e",{text:" "+"to retrieve your issues"})),h=!0)}h||d.text(AJS.format(d.text(),e.status));jQuery(".calculating, .result, .data",c).remove()}})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.synchrony-interop:synchrony-status-banner-loader', location = '/js/synchrony-down-banner-loader.js' */
require(["ajs","wrm"],function(b,a){b.toInit(function(){!0===a.data.claim("com.atlassian.confluence.plugins.synchrony-interop:synchrony-status-banner-loader.synchrony-status")&&a.require("wrc!synchrony-interop-down-banner")})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.recently-viewed-plugin:app-resources', location = 'templates/recently.soy' */
// This file was automatically generated from recently.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace RY.Templates.
 */

if (typeof RY == 'undefined') { var RY = {}; }
if (typeof RY.Templates == 'undefined') { RY.Templates = {}; }


RY.Templates.body = function(opt_data, opt_ignored) {
  return '<div id="recently-viewed" class="aui-group"><div class="aui-item"><div class="top"><div class="filter"><form class="aui"><input class="filter-input text" type="text" placeholder="' + soy.$$escapeHtmlAttribute('Filter') + '"></form></div></div><div class="pages"></div></div></div>';
};
if (goog.DEBUG) {
  RY.Templates.body.soyTemplateName = 'RY.Templates.body';
}


RY.Templates.pageCollection = function(opt_data, opt_ignored) {
  return '<div class="groups"></div><div class="empty"></div>';
};
if (goog.DEBUG) {
  RY.Templates.pageCollection.soyTemplateName = 'RY.Templates.pageCollection';
}


RY.Templates.pageGroup = function(opt_data, opt_ignored) {
  return '<h3>' + soy.$$escapeHtml(opt_data.title) + '</h3><ul></ul>';
};
if (goog.DEBUG) {
  RY.Templates.pageGroup.soyTemplateName = 'RY.Templates.pageGroup';
}


RY.Templates.pageItem = function(opt_data, opt_ignored) {
  return '<a href=' + soy.$$escapeHtmlAttributeNospace(soy.$$filterNormalizeUri(opt_data.href)) + '><div class="page-icon"><span class="aui-icon content-type-' + soy.$$escapeHtmlAttribute(opt_data.type) + '">' + soy.$$escapeHtml(opt_data.type) + '</span></div><div class="page-title"><span class="ellipsis" href=' + soy.$$escapeHtmlAttributeNospace(soy.$$filterNormalizeUri(opt_data.href)) + '>' + soy.$$escapeHtml(opt_data.title) + ' - ' + soy.$$escapeHtml(opt_data.space) + '</span></div></a>';
};
if (goog.DEBUG) {
  RY.Templates.pageItem.soyTemplateName = 'RY.Templates.pageItem';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.recently-viewed-plugin:app-resources', location = 'js/util.js' */
var RY=RY||{};
(function(){var a=new Date,c=(new Date(a)).getTime(),d=(new Date(a.getFullYear(),a.getMonth(),a.getDate())).getTime(),h=(new Date(a.getFullYear(),a.getMonth(),a.getDate()-1)).getTime();RY.util=RY.Util={analytics:{trackDialogOpen:function(){AJS.trigger("analytics",{name:"recently-viewed-dialog-open"})},trackPageOpen:function(){AJS.trigger("analytics",{name:"recently-viewed-page-open"})}},quote:function(b){return b.replace(/([.?*+^$[\]\\(){}|-])/g,"\\$1")},diffInDays:function(b,a){return Math.floor((b-a)/
864E5)},daysSince:function(a){return a>=d?0:a>=h?1:RY.util.diffInDays(c,a)},wait:function(a,e,c){var b,f,g,d=function(){f=setTimeout(function(){a.apply(c,g)},e)};return function(){g=arguments;var a=new Date;b&&a-b<e&&clearTimeout(f);d();b=a}}}})();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.recently-viewed-plugin:app-resources', location = 'js/page-storage.js' */
var RY=RY||{};
RY.RecentPageStorage=function(d){var c={},e=function(){var a=function(b,a){return"lastSeen"===b&&_.isString(a)?(new Date(a)).getTime():a};try{c=JSON.parse(d.getItem("com.atlassian.confluence.plugins.recently.viewed.pages.v1"),a)||{}}catch(b){c={}}return c},f=function(){var a=_.values(c),b=a.length-100;if(0<b){var d=_.sortBy(a,function(a){return a.lastSeen});_.times(b,function(){var a=d.shift();delete c[a.id]})}};this.addCurrentPage=function(a){if(a.id&&a.title){e();var b=c[a.id];b||(c[a.id]=b={});
b=_.extend(b,a);b.lastSeen=(new Date).getTime();b.count=b.count+1||1;f();try{d.setItem("com.atlassian.confluence.plugins.recently.viewed.pages.v1",JSON.stringify(c))}catch(g){}}};this.getPages=function(){return _.values(e())}};
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.recently-viewed-plugin:app-resources', location = 'js/filter.js' */
var RY=RY||{};
RY.FilterView=Backbone.View.extend({className:"filter",events:{"input .filter-input":"onInput","keyup .filter-input":"onInput","keydown .filter-input":"onKeydown"},initialize:function(){_.bindAll(this,"render","onInput","onKeydown","search");this.navigationEvents=this.options.navigationEvents;this.onInput=RY.util.wait(this.onInput,100,this)},render:function(){this.$input=this.$(".filter-input");return this},onInput:function(a){a&&_.contains([37,38,39,40],a.which)||this.search()},onKeydown:function(a){switch(a.which){case 13:this.navigationEvents.trigger("select");a.preventDefault();
a.stopPropagation();break;case 38:this.navigationEvents.trigger("previous");a.preventDefault();break;case 40:this.navigationEvents.trigger("next"),a.preventDefault()}},search:function(){var a=this.$input.val();this.collection.search(a)}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.recently-viewed-plugin:app-resources', location = 'js/page-model.js' */
var RY=RY||{};
(function(){var c="undefined"!=typeof ConfluenceMobile?ConfluenceMobile.AppData.get("confluence-context-path"):AJS.contextPath();RY.Page=Backbone.Model.extend({href:function(){return c+this.get("url")},daysSinceLastSeen:function(){return RY.util.daysSince(this.get("lastSeen"))}});RY.PageCollection=Backbone.Collection.extend({model:RY.Page,url:c+"/rest/recentlyviewed/1.0/recent?includeTrashedContent\x3dtrue",search:function(a){if(0===this._queryLength(a))var b=this.models;else{var c=a.match(/[^\s-]+/g);
b=this.filter(function(a){var b=a.get("title");a=a.get("space");var d=(b+a).toLowerCase();return _.all(c,function(a){return-1!==d.indexOf(a.toLowerCase())})})}this.trigger("filter",b,a);return b},_queryLength:function(a){return String.prototype.trim?a.trim().length:AJS.$.trim(a).length},comparator:function(a){return-a.get("lastSeen")}})})();
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.recently-viewed-plugin:app-resources', location = 'js/page.js' */
var RY=RY||{};RY.PageView=Backbone.View.extend({tagName:"li",className:"ry-page",template:RY.Templates.pageItem,initialize:function(){_.bindAll(this,"hide","show","render")},hide:function(){this.$el.hide().removeClass("shown")},show:function(){this.$el.show().addClass("shown")},render:function(){var a=this.model.toJSON();a.href=this.model.href();this.$el.html(this.template(a));return this}});
RY.PageGroupView=Backbone.View.extend({className:"group",template:RY.Templates.pageGroup,initialize:function(){_.bindAll(this,"hide","hideAll","show","showBorder","showPages","add","render");this.title=this.options.title;this.modelViews={}},hide:function(){this.$el.hide()},hideAll:function(){this.$el.removeClass("border").hide();_.invoke(_.values(this.modelViews),"hide")},show:function(){this.$el.show()},showBorder:function(){this.$el.addClass("border")},showPages:function(a){var b=this,c=!1;_.each(a,
function(a){if(a=b.modelViews[a.cid])c=!0,a.show()});c&&this.show();return c},add:function(a){var b=new RY.PageView({model:a});this.modelViews[a.cid]=b;this.$list.append(b.render().el)},render:function(){this.$el.html(this.template({title:this.title}));this.$list=this.$("ul");return this}});
RY.PageNavigator=function(a,b,c){function e(b){pageItems=a.find("li.shown");var c=pageItems.index(pageItems.filter(".highlight"));d&&d.removeClass("highlight");b+=c;0>b&&(b=pageItems.length-1);b>=pageItems.length&&(b=0);d=pageItems.eq(b);d.addClass("highlight")}function f(){if(d.length){var c=b.children(),e=b.height(),f=d.outerHeight(!0),g=d.position().top;0>g?0===a.find("li.shown").index(d)?b.scrollTop(0):b.scrollTop(d.offset().top-c.offset().top):g>e&&b.scrollTop(d.offset().top-c.offset().top-e+
f)}}var d=null;c.on("select",function(){if(d&&d.is(":visible")){RY.util.analytics.trackPageOpen();var a=d.find(".page-title span").attr("href");window.location=a}},this);c.on("previous",function(){e(-1);f()},this);c.on("next",function(){e(1);f()},this)};
RY.PageCollectionView=Backbone.View.extend({template:RY.Templates.pageCollection,events:{"click .page-title a":RY.util.analytics.trackPageOpen},initialize:function(){_.bindAll(this,"checkEmpty","filter","_groupForPage","addOne","showEmptyMessage","clearEmptyMessage","addAll","render");this.modelViews={};this.collection.on("reset",this.addAll,this);this.collection.on("add",this.addOne,this);this.collection.on("filter",this.filter,this)},checkEmpty:function(a,b){var c=this.collection.isEmpty();a=0===
a.length;c||a?(c?b="Sorry mate, looks like you haven\u0027t visited any pages yet.":(b=AJS.contextPath()+"/dosearchsite.action?queryString\x3d"+encodeURIComponent(b),b="We didn\u0027t find any matching pages in your history."+" "+AJS.format("\u003ca href=\u0022{0}\u0022\u003eClick here\u003c/a\u003e to search for this term instead.",b)),this.showEmptyMessage(b)):this.clearEmptyMessage()},filter:function(a,b){this.checkEmpty(a,b||"");b=[this.$today,this.$yesterday,this.$older];_.invoke(b,
"hideAll");var c=[];_.each(b,function(b){b.showPages(a)&&c.push(b)});1<c.length&&(c.pop(),_.invoke(c,"showBorder"))},_groupForPage:function(a){a=a.daysSinceLastSeen();return 0===a?this.$today:1===a?this.$yesterday:this.$older},addOne:function(a){this._groupForPage(a).add(a)},showEmptyMessage:function(a){this.$(".empty").html(AJS.$("\x3cp\x3e").html(a))},clearEmptyMessage:function(){this.$(".empty").html("")},addAll:function(){this.collection.each(this.addOne)},render:function(){this.$el.html(this.template());
this.$today=new RY.PageGroupView({title:"Today"});this.$yesterday=new RY.PageGroupView({title:"Yesterday"});this.$older=new RY.PageGroupView({title:"Older"});var a=this.$(".groups");a.append(this.$today.render().el);a.append(this.$yesterday.render().el);a.append(this.$older.render().el);
_.invoke([this.$today,this.$yesterday,this.$older],"hideAll");return this}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.recently-viewed-plugin:main-resources', location = 'js/main.js' */
var RY=RY||{};
AJS.toInit(function(b){b("#view-user-history-link").unbind("click");b("#view-user-history-link").click(function(a){a.preventDefault();a=new AJS.Dialog({width:600,height:500,id:"recently-viewed-dialog",closeOnOutsideClick:!0});var c=b(RY.Templates.body());a.addHeader("Recently viewed pages");a.addPanel("SinglePanel",c);a.addLink("Close",function(a){a.hide()});var f=b("\x3cdiv\x3e",{"class":"dialog-tip"}).text("Hint: type \u0022g\u0022 and then \u0022r\u0022 anywhere to quickly open this menu");
a.popup.element.find(".dialog-button-panel").append(f);var d=new RY.PageCollection,e=new RY.PageCollectionView({collection:d});c.find(".pages").html(e.render().el);a.gotoPanel(0);a.show();var g=b("#recently-viewed-dialog").spin("large");d.fetch({success:function(){g.spinStop();var a=_.extend({},Backbone.Events);new RY.PageNavigator(e.$el,c.parent(),a);(new RY.FilterView({collection:d,el:c.find(".filter"),navigationEvents:a})).render().search()}});RY.util.analytics.trackDialogOpen()})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacemenu-resources', location = 'js/space-menu.js' */
AJS.toInit(function(a){var b=a(Confluence.Templates.BrowseSpaces.spacesLink());a("#space-directory-link").replaceWith(a(b));var d=function(b){var c=a("#space-menu-link-content");c.html(b.template);AJS&&AJS.Confluence&&AJS.Confluence.Analytics&&AJS.Confluence.Analytics.setAnalyticsSource&&AJS.Confluence.Analytics.setAnalyticsSource(c.find("a"),"spacemenu");a("#create-space-header").click(function(){AJS.trigger("analytics",{name:"create-space-from-header"});Confluence.SpaceBlueprint.Dialog.launch();
return!1})};a("#space-menu-link-content").on("aui-dropdown2-show",function(){AJS.trigger("analytics",{name:"open-space-menu"});a("#space-menu-link-content .aui-dropdown2-section")&&a("#space-menu-link-content .aui-dropdown2-section").length||a.ajax({url:Confluence.getContextPath()+"/rest/ia/1.0/spacesmenu",type:"GET",dataType:"json",cache:!1,success:d});return!1})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:spacemenu-resources', location = 'soy/space-menu.soy' */
// This file was automatically generated from space-menu.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.BrowseSpaces.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.BrowseSpaces == 'undefined') { Confluence.Templates.BrowseSpaces = {}; }


Confluence.Templates.BrowseSpaces.spacesLink = function(opt_data, opt_ignored) {
  return '' + aui.dropdown2.trigger({menu: {id: 'space-menu-link-content'}, id: 'space-menu-link', tagName: 'a', extraClasses: 'aui-nav-link', title: 'Spaces', showIcon: false, text: 'Spaces'}) + aui.dropdown2.contents({id: 'space-menu-link-content', extraClasses: 'aui-style-default aui-dropdown2-in-header'});
};
if (goog.DEBUG) {
  Confluence.Templates.BrowseSpaces.spacesLink.soyTemplateName = 'Confluence.Templates.BrowseSpaces.spacesLink';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-space-ia:space-ia-analytics', location = 'js/space-ia-analytics.js' */
(function(f){function d(){var a=f(".acs-side-bar"),b=a.find("a:not(.external_link a, #acs-configure-link)");c(b,"sidebar");b=a.find("[data-collapsed-tooltip\x3d'"+AJS.I18n.getText("Pages")+"']");c(b,"sidebar-pages");b=a.find("[data-collapsed-tooltip\x3d'"+AJS.I18n.getText("Blog")+"']");c(b,"sidebar-blogs");b=a.find(".quick-links-section li:not(.external_link) a");c(b,"spaceshortcut");b=a.find(".ia-secondary-container a:not(.more-children-link)");"pages"==a.find(".ia-secondary-container").data("tree-type")?
c(b,"contextnavchildmode"):"page-tree"==a.find(".ia-secondary-container").data("tree-type")?c(b,"contextnavpagetreemode"):c(b,"contextnav")}function e(a){return function(){AJS.trigger("analytics",{name:"space-ia-nav",data:{linkType:a}})}}var c=AJS.Confluence.Analytics.setAnalyticsSource;AJS.toInit(function(a){a(".acs-side-bar").find(".ia-secondary-container .more-children-link").click(e("context-nav.more-children"));AJS.bind("sidebar.exit-configure-mode",d);AJS.bind("sidebar.flyout-triggered",function(b,
a){e("flyout-triggered."+a.flyout)()});AJS.bind("pagetree-children-loaded",d);d()})})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.aui.staging:persistable', location = 'js/persistable.js' */
define("confluence/persistable",["skate","confluence/storage-manager"],function(f,g){var d=g("confluence","userSettings");return f("data-persist",{type:f.types.ATTRIBUTE,created:function(b){if(!b.name)throw'Missing attribute "name"';var c=b.getAttribute("data-persist"),a=b.getAttribute("data-persist-scope"),a=a?b.name+"."+a.replace(/\./g,"-"):b.name,a=d.getItem(a),e=b.getAttribute("data-default-value");if(/value|checked/.test(c))a=a||e||null,null!==a&&("checked"===c&&(a="true"===a),b[c]=a);else throw"Persistable only works with 'value' or 'checked' attributes!";
},events:{change:function(b){var c=b.getAttribute("data-persist"),a=b.getAttribute("data-persist-scope"),a=a?b.name+"."+a.replace(/\./g,"-"):b.name,e=b.getAttribute("data-default-value");String(b[c])===e?d.removeItem(a):d.setItem(a,b[c],2592E3)}}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-quicknav:quicknav-resources', location = 'js/quick-nav.js' */
define("confluence-quicknav/quick-nav",["jquery","confluence/api/logger"],function(b,l){var f=[],d,g=function(a){return function(e){a.closest("form").find(".quick-nav-drop-down").append(e)}},h=function(a){"undefined"!==typeof a?f.push(a):l.log("WARN: Attempted to add a dropdown post-process function that was undefined.")},m=function(a){b("a",a).each(function(){var a=b(this),c=a.find("span").data("properties");(c=c?c.spaceName:null)&&!a.is(".content-type-spacedesc")&&(a.after(a.clone().attr("class",
"space-name").html(c)),a.parent().addClass("with-space-name"))})};return{addDropDownPostProcess:h,setMakeParams:function(a){d=a},init:function(){var a=b("#quick-search-query"),e=g(a),c=b("#space-blog-quick-search-query"),k=b("#confluence-space-key");a.quicksearch("/rest/quicknav/1/search",null,{dropdownPlacement:e,dropdownPostprocess:function(a){b.each(f,function(c,b){b&&b(a)})},makeParams:function(a){return d?d(a):{query:a}},ajsDropDownOptions:{className:"quick-search-dropdown"}});h(m);c.length&&
k.length&&c.quicksearch("/rest/quicknav/1/search?type\x3dblogpost\x26spaceKey\x3d"+b("\x3ci\x3e\x3c/i\x3e").html(k.attr("content")).text(),null,{dropdownPlacement:g(c)})}}});require("confluence/module-exporter").exportModuleAsGlobal("confluence-quicknav/quick-nav","Confluence.QuickNav");
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-license-banner:confluence-license-banner-resources', location = '/js/confluence-license-banner.js' */
require(["ajs","jquery"],function(a,c){a.toInit(function(){var e=WRM.data.claim("com.atlassian.confluence.plugins.confluence-license-banner:confluence-license-banner-resources.license-details");if(e){var d={destroyBanner:function(){c("#license-banner").slideUp(function(){c("#license-banner").remove()})},remindMeLater:function(){c.ajax({type:"POST",dataType:"json",contentType:"application/json; charset\x3dutf-8",data:JSON.stringify({atl_token:a.Meta.get("atl-token")}),url:a.contextPath()+"/rest/licensebanner/1.0/reminder/later"});
d.destroyBanner()},remindMeNever:function(){c.ajax({type:"POST",dataType:"json",contentType:"application/json; charset\x3dutf-8",data:JSON.stringify({atl_token:a.Meta.get("atl-token")}),url:a.contextPath()+"/rest/licensebanner/1.0/reminder/never"});d.destroyBanner()},createBanner:function(b){var a;b.showLicenseExpiryBanner?a=Confluence.Templates.LicenseBanner.expiryBanner({days:b.daysBeforeLicenseExpiry,mac:b.renewUrl,sales:b.salesUrl}):b.showMaintenanceExpiryBanner&&(a=Confluence.Templates.LicenseBanner.maintenanceBanner({days:b.daysBeforeMaintenanceExpiry,
mac:b.renewUrl,sales:b.salesUrl}));a&&(b=c(a),c("#full-height-container").prepend(b),c("#license-banner-later").click(function(a){a.preventDefault();d.remindMeLater()}),c("#license-banner-never").click(function(a){a.preventDefault();d.remindMeNever()}),b.find(".icon-close").click(function(a){a.preventDefault();d.remindMeLater()}))}};d.createBanner(e)}else a.warn("Unable to claim my required data")})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-license-banner:confluence-license-banner-resources', location = 'soy/confluence-license-banner.soy' */
// This file was automatically generated from confluence-license-banner.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.LicenseBanner.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.LicenseBanner == 'undefined') { Confluence.Templates.LicenseBanner = {}; }


Confluence.Templates.LicenseBanner.wrapper = function(opt_data, opt_ignored) {
  return '<div>' + soy.$$filterNoAutoescape(opt_data.contentHTML) + '</div>';
};
if (goog.DEBUG) {
  Confluence.Templates.LicenseBanner.wrapper.soyTemplateName = 'Confluence.Templates.LicenseBanner.wrapper';
}


Confluence.Templates.LicenseBanner.expiryBanner = function(opt_data, opt_ignored) {
  var output = '';
  var renewTag__soy8 = '<a id="license-banner-my-link" target="_blank" href="' + soy.$$escapeHtml(opt_data.mac) + '">';
  var end__soy12 = '</a>';
  var mailTag__soy14 = '<a id="license-banner-sales-link" target="_blank" href="' + soy.$$escapeHtml(opt_data.sales) + '">';
  output += Confluence.Templates.LicenseBanner.wrapper({contentHTML: '' + aui.message.warning({content: '<div id="license-banner-content" data-days="' + soy.$$escapeHtml(opt_data.days) + '" data-subscription="true">' + ((opt_data.days < 0) ? soy.$$filterNoAutoescape(AJS.format('Your subscription has expired. This site is now read-only. To keep using Confluence, {0}renew online{1} or {2}contact us{3}.',renewTag__soy8,end__soy12,mailTag__soy14,end__soy12)) : (opt_data.days == 0) ? soy.$$filterNoAutoescape(AJS.format('Your subscription will expire today and your site will become read-only. To keep using Confluence, {0}renew online{1} or {2}contact us{3}.',renewTag__soy8,end__soy12,mailTag__soy14,end__soy12)) : soy.$$filterNoAutoescape(AJS.format('Your subscription will expire in {0,choice,1#{0} day|1\x3c{0} days}. Your site will become read-only after the expiry. To keep using Confluence, {1}renew online{2} or {3}contact us{4}.',opt_data.days,renewTag__soy8,end__soy12,mailTag__soy14,end__soy12))) + '</div>', isCloseable: opt_data.days > 7, id: 'license-banner'})});
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.LicenseBanner.expiryBanner.soyTemplateName = 'Confluence.Templates.LicenseBanner.expiryBanner';
}


Confluence.Templates.LicenseBanner.maintenanceBanner = function(opt_data, opt_ignored) {
  var output = '';
  var renewTag__soy39 = '<a id="license-banner-my-link" target="_blank" href="' + soy.$$escapeHtml(opt_data.mac) + '">';
  var end__soy43 = '</a>';
  var later__soy45 = '<a href="#" id="license-banner-later">';
  var never__soy47 = '<a href="#" id="license-banner-never">';
  output += Confluence.Templates.LicenseBanner.wrapper({contentHTML: '' + aui.message.warning({content: '<div id="license-banner-content" data-days="' + soy.$$escapeHtml(opt_data.days) + '" data-subscription="false">' + ((opt_data.days < 0) ? soy.$$filterNoAutoescape(AJS.format('You no longer have access Support and Updates. {0}Renew online{1}, {2}remind me later{3} or {4}never remind me again{5}.',renewTag__soy39,end__soy43,later__soy45,end__soy43,never__soy47,end__soy43)) : (opt_data.days == 0) ? soy.$$filterNoAutoescape(AJS.format('You will lose access to Support and Updates today. {0}Renew online{1}, {2}remind me later{3} or {4}never remind me again{5}.',renewTag__soy39,end__soy43,later__soy45,end__soy43,never__soy47,end__soy43)) : soy.$$filterNoAutoescape(AJS.format('You will lose access to Support and Updates in {0,choice,1#{0} day|1\x3c{0} days}. {1}Renew online{2}, {3}remind me later{4} or {5}never remind me again{6}.',opt_data.days,renewTag__soy39,end__soy43,later__soy45,end__soy43,never__soy47,end__soy43))) + '</div>', id: 'license-banner'})});
  return output;
};
if (goog.DEBUG) {
  Confluence.Templates.LicenseBanner.maintenanceBanner.soyTemplateName = 'Confluence.Templates.LicenseBanner.maintenanceBanner';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-browser-metrics:legacy-browser-metrics', location = '/js/legacy/browser-metrics.js' */
(function(){var a=window.BrowserMetrics||{};a.isFunction=function(b){return !!(b&&b.constructor&&b.call&&b.apply)};a.isEnabled=function(){if(a.enabled===undefined){a.enabled=true}return a.enabled};window.BrowserMetrics=a}());(function(){var b=window.BrowserMetrics||{};if(b.isEnabled()){var f=5;var e=12000;var c=function(g){return Math.round(g)};var a=function(g){return Math.round(g*100)/100};var d=function(k){var h=(function(){var n=/^(\w+):\/\/([^\/]*)(.*)$/;return function(p){var o=p.match(n);if(!o){return{path:p}}return{scheme:o[1],host:o[2],path:o[3]}}}());var i=(function(){var n=["secureConnectionStart","requestStart","responseStart","responseEnd","domContentLoadedEventStart","domContentLoadedEventEnd","loadEventEnd"];return function(o){if(k.performance){var s=k.performance.timing;var p=s.navigationStart;if(p){for(var r=0;r<n.length;++r){var q=n[r];var t=s[q];if(t){o(q,t-p)}}}}}}());var g=(function(){var o=[{key:"LOGIN",pattern:/^\/login.*/i},{key:"J-DASH",pattern:/^\/secure\/dashboard\.jspa.*/i},{key:"J-ISSUE",pattern:/^\/browse\/\w+\-\w+.*/i},{key:"J-NAV",pattern:/^\/issues.*/i},{key:"J-RAPID",pattern:/secure\/rapidboard\.jspa/i},{key:"SD-AGENT",pattern:/^(\/\w+)?\/servicedesk\/agent\/.*/i},{key:"SD-CUSTOMER",pattern:/^(\/\w+)?\/servicedesk\/customer\/.*/i},{key:"C-DASH",pattern:/^\/wiki(\/)?(\?.*|#.*)?$/i},{key:"C-DASH",pattern:/^\/wiki\/dashboard\.action.*$/i},{key:"C-SPACE",pattern:/^\/wiki\/display\/\w+(\?.*|#.*)?$/i},{key:"C-PAGE",pattern:/^\/wiki\/display\/\w+\/.*/i},{key:"C-PAGE",pattern:/^\/wiki\/pages\/viewpage\.action.*/i},{key:"C-BLOG",pattern:/^\/wiki\/display\/~\w+\/\d+\/\d+\/\d+\/.*/i},{key:"C-EDITOR",pattern:/^\/wiki\/pages\/editpage\.action.*/i},{key:"C-CREATE",pattern:/^\/wiki\/pages\/createpage\.action.*/i}];return function n(){var r=h(k.location.href).path;for(var p=0;p<o.length;++p){var q=o[p];if(r.match(q.pattern)){return q.key}}return null}}());function j(){var n=g();if(n){i(function(p,r){var o="browser.metrics."+p,q={version:f,page:n,value:r>e?"x":Math.ceil((r)/100),rawValue:c(r)};AJS.Analytics?AJS.Analytics.triggerPrivacyPolicySafeEvent(o,q):AJS.trigger("analyticsEvent",{name:o,data:q})})}}function m(){try{j()}catch(n){if(window.console){window.console.log("Error reporting browser metrics: "+n)}}}function l(){if(k.performance.timing.loadEventEnd){m()}else{setTimeout(l,1000)}}if(k.performance&&k.performance.timing){l()}};if(!window.ATL_PERF){window.ATL_PERF={}}window.ATL_PERF.initPageLoad=d}}());
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-browser-metrics:legacy-browser-metrics', location = '/js/legacy/browser-metrics-events.js' */
(function(){var b=window.BrowserMetrics||{};if(b.isEnabled()){var e=5;var d=12000;var a=function(f){return Math.round(f*100)/100};var c=function(g){var l={};function h(){return g.performance&&g.performance.now?g.performance.now():new Date().getTime()}function n(o){return o===Object(o)}function k(o){if(n(o)){return o.eventName+"."+o.eventType}else{return o}}function m(o){if(n(o)){return o.eventName}else{return o}}function j(o){if(n(o)){return o.eventType}else{return""}}function f(o){var p=k(o);l[p]=h()}function i(r,u){var v=k(r);if(!l[v]){throw ("Error logging browser metrics event end: no start event for key '"+v+"'")}var t=h()-l[v];l[v]=null;var o=m(r),q=j(r);var p="browser.metrics.e."+o+(u?"."+u:""),s={version:e,value:t>d?"x":Math.ceil((t)/100),rawValue:a(t),eventType:q};g.AJS.Analytics?g.AJS.Analytics.triggerPrivacyPolicySafeEvent(p,s):g.AJS.trigger("analyticsEvent",{name:p,data:s})}return{start:f,end:i}};if(!window.ATL_PERF){window.ATL_PERF={}}window.ATL_PERF.initEvents=c}}());
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-browser-metrics:legacy-browser-metrics', location = '/js/legacy/browser-metrics-init.js' */
(function(){var a=window.BrowserMetrics||{};if(a.isEnabled()){if(Math.random()<0.1){window.ATL_PERF&&window.ATL_PERF.initPageLoad&&window.ATL_PERF.initPageLoad(window)}window.ATL_PERF&&window.ATL_PERF.initEvents&&(function(){window.BrowserMetrics=window.ATL_PERF.initEvents(window)}())}}());
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:rotp-menu', location = 'appswitcher/appswitcher.soy' */
// This file was automatically generated from appswitcher.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace navlinks.templates.appswitcher.
 */

if (typeof navlinks == 'undefined') { var navlinks = {}; }
if (typeof navlinks.templates == 'undefined') { navlinks.templates = {}; }
if (typeof navlinks.templates.appswitcher == 'undefined') { navlinks.templates.appswitcher = {}; }


navlinks.templates.appswitcher.linkSection = function(opt_data, opt_ignored) {
  var output = '';
  if (opt_data.list.length > 0) {
    output += '<div class="aui-nav-heading sidebar-section-header">' + soy.$$escapeHtml(opt_data.title) + '</div><ul class="aui-nav nav-links">';
    var linkList8 = opt_data.list;
    var linkListLen8 = linkList8.length;
    for (var linkIndex8 = 0; linkIndex8 < linkListLen8; linkIndex8++) {
      var linkData8 = linkList8[linkIndex8];
      output += navlinks.templates.appswitcher.applicationsItem(linkData8);
    }
    output += '</ul>';
  }
  return output;
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.linkSection.soyTemplateName = 'navlinks.templates.appswitcher.linkSection';
}


navlinks.templates.appswitcher.applicationsItem = function(opt_data, opt_ignored) {
  return '<li class="nav-link"><a href="' + soy.$$escapeHtml(opt_data.link) + '" ' + ((opt_data.self) ? 'class="checked"' : '') + ' title="' + soy.$$escapeHtml(opt_data.link) + '"><span class="nav-link-label">' + soy.$$escapeHtml(opt_data.label) + '</span></a></li>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.applicationsItem.soyTemplateName = 'navlinks.templates.appswitcher.applicationsItem';
}


navlinks.templates.appswitcher.shortcutsItem = function(opt_data, opt_ignored) {
  return '<li class="nav-link"><a href="' + soy.$$escapeHtml(opt_data.link) + '" ' + ((opt_data.self) ? 'class="checked"' : '') + ' title="' + soy.$$escapeHtml(opt_data.link) + '"><span class="nav-link-label">' + soy.$$escapeHtml(opt_data.label) + '</span>' + ((opt_data.showDescription && opt_data.description) ? '<span class="nav-link-description">' + soy.$$escapeHtml(opt_data.description) + '</span>' : '') + '</a></li>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.shortcutsItem.soyTemplateName = 'navlinks.templates.appswitcher.shortcutsItem';
}


navlinks.templates.appswitcher.error = function(opt_data, opt_ignored) {
  return '<div class="app-switcher-error">' + soy.$$filterNoAutoescape('Something went wrong, please \x3cspan class\x3d\x22app-switcher-retry\x22\x3etry again\x3c/span\x3e.') + '</div>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.error.soyTemplateName = 'navlinks.templates.appswitcher.error';
}


navlinks.templates.appswitcher.sidebarContents = function(opt_data, opt_ignored) {
  return '<div class="aui-page-panel-nav"><nav class="aui-navgroup aui-navgroup-vertical"><div class="app-switcher-section app-switcher-applications"><div class="aui-nav-heading">' + soy.$$escapeHtml('Application Links') + '</div><div class="app-switcher-loading">' + soy.$$filterNoAutoescape('Loading...') + '</div></div><div class="app-switcher-section app-switcher-shortcuts"><div class="aui-nav-heading">' + soy.$$escapeHtml('Shortcuts') + '</div><div class="app-switcher-loading">' + soy.$$filterNoAutoescape('Loading...') + '</div></div></nav></div>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.sidebarContents.soyTemplateName = 'navlinks.templates.appswitcher.sidebarContents';
}


navlinks.templates.appswitcher.trigger = function(opt_data, opt_ignored) {
  return '<span class="aui-icon aui-icon-small aui-iconfont-appswitcher">' + soy.$$escapeHtml('Linked Applications') + '</span>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.trigger.soyTemplateName = 'navlinks.templates.appswitcher.trigger';
}


navlinks.templates.appswitcher.projectHeaderSection = function(opt_data, opt_ignored) {
  return '<div class="app-switcher-title">' + aui.avatar.avatar({size: 'large', avatarImageUrl: opt_data.avatarUrl, isProject: true, title: opt_data.name}) + '<div class="sidebar-project-name">' + soy.$$escapeHtml(opt_data.name) + '</div></div>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.projectHeaderSection.soyTemplateName = 'navlinks.templates.appswitcher.projectHeaderSection';
}


navlinks.templates.appswitcher.cogDropdown = function(opt_data, opt_ignored) {
  var output = '';
  var dropdownList__soy74 = '' + navlinks.templates.appswitcher.dropdownList({list: opt_data.links});
  output += aui.dropdown2.dropdown2({menu: {id: opt_data.id, content: dropdownList__soy74, extraClasses: 'aui-style-default sidebar-customize-section'}, trigger: {showIcon: false, content: '<span class="aui-icon aui-icon-small aui-iconfont-configure"></span>', container: '#app-switcher'}});
  return output;
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.cogDropdown.soyTemplateName = 'navlinks.templates.appswitcher.cogDropdown';
}


navlinks.templates.appswitcher.dropdownList = function(opt_data, opt_ignored) {
  var output = '<ul class="sidebar-admin-links">';
  var linkList82 = opt_data.list;
  var linkListLen82 = linkList82.length;
  for (var linkIndex82 = 0; linkIndex82 < linkListLen82; linkIndex82++) {
    var linkData82 = linkList82[linkIndex82];
    output += '<li class="nav-link"><a href="' + soy.$$escapeHtml(linkData82.href) + '" title="' + soy.$$escapeHtml(linkData82.title) + '"><span class="nav-link-label">' + soy.$$escapeHtml(linkData82.label) + '</span></a></li>';
  }
  output += '</ul>';
  return output;
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.dropdownList.soyTemplateName = 'navlinks.templates.appswitcher.dropdownList';
}


navlinks.templates.appswitcher.switcher = function(opt_data, opt_ignored) {
  var output = '';
  if (true) {
    if (AJS.DarkFeatures.isEnabled('rotp.sidebar')) {
      var sidebarContents__soy97 = '' + navlinks.templates.appswitcher.sidebarContents(null);
      var triggerContent__soy99 = '' + navlinks.templates.appswitcher.trigger(null);
      output += navlinks.templates.appswitcher.sidebar({sidebar: {id: 'app-switcher', content: sidebarContents__soy97}, trigger: {showIcon: false, content: triggerContent__soy99}});
    } else {
      output += navlinks.templates.appswitcher_old.switcher(null);
    }
  }
  return output;
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.switcher.soyTemplateName = 'navlinks.templates.appswitcher.switcher';
}


navlinks.templates.appswitcher.sidebar = function(opt_data, opt_ignored) {
  return '<button class="sidebar-trigger app-switcher-trigger" aria-owns="' + soy.$$escapeHtml(opt_data.sidebar.id) + '" aria-haspopup="true" >' + soy.$$filterNoAutoescape(opt_data.trigger.content) + '</button><div id=' + soy.$$escapeHtml(opt_data.sidebar.id) + ' class="app-switcher-sidebar aui-style-default sidebar-offscreen" data-is-sidebar="true" data-is-user-admin="' + soy.$$escapeHtml(false) + '">' + soy.$$filterNoAutoescape(opt_data.sidebar.content) + '</div>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher.sidebar.soyTemplateName = 'navlinks.templates.appswitcher.sidebar';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:rotp-menu', location = 'appswitcher/appswitcher.js' */
(function(e,a){var d="is-user-admin";var c="#app-switcher";a.SideBar=function(f){var g=this;this.$sidebar=null;f=e.extend({sidebarContents:null},f);this.getLinks=function(){return e.ajax({url:AJS.contextPath()+"/rest/menu/latest/appswitcher",cache:false,dataType:"json"}).done(this.updateAppLinks).fail(this.showAppSwitcherError)};this.populateProjectHeader=function(i,h){g.getSidebar().find(".app-switcher-shortcuts .aui-nav-heading").after(navlinks.templates.appswitcher.projectHeaderSection({avatarUrl:h,name:i}))};this.getProjectData=function(){var h=e(".project-shortcut-dialog-trigger"),i=h.data("key"),j=h.data("entity-type");if(h.size()==0||!i||!j){e(".app-switcher-shortcuts").remove();return}var l,k;k=e.ajax({url:AJS.contextPath()+"/rest/project-shortcuts/1.0/local/"+i,cache:false,data:{entityType:j},dataType:"json"});l=e.ajax({url:AJS.contextPath()+"/rest/project-shortcuts/1.0/remote/"+i,cache:false,data:{entityType:j},dataType:"json"});e.when(k,l).then(function(n,m){g.updateProjectShortcuts(n,m,{key:i,entityType:j,name:h.data("name"),avatarUrl:h.find("img").prop("src")})},g.showProjectShortcutsError)};this.getSidebar=function(){if(!this.$sidebar){this.$sidebar=e(f.sidebarContents)}return this.$sidebar};this.addApplicationsCog=function(){e(".app-switcher-applications .aui-nav-heading").before(navlinks.templates.appswitcher.cogDropdown({id:"sidebar-applications-admin-dropdown",links:[{href:AJS.contextPath()+"/plugins/servlet/customize-application-navigator",label:"Customize navigator",title:"Add new entries, hide existing or restrict who sees what"},{href:AJS.contextPath()+"/plugins/servlet/applinks/listApplicationLinks",label:"Manage application links",title:"Link to more Atlassian applications"}]}))};this.addProjectShortcutsCog=function(h,j){var i=[{href:AJS.contextPath()+"/plugins/servlet/custom-content-links-admin?entityKey="+h,label:"Customize shortcuts",title:""}];if(g.entityMappings[j]){i.push({href:g.generateEntityLinksUrl(h,g.entityMappings[j]),label:"Manage product links",title:""})}g.getSidebar().find(".app-switcher-shortcuts .aui-nav-heading").before(navlinks.templates.appswitcher.cogDropdown({id:"sidebar-project-shortcuts-admin-dropdown",links:i}))};this.updateAppLinks=function(h){e(function(){g.getSidebar().find(".app-switcher-applications").html(navlinks.templates.appswitcher.linkSection({title:"Application Links",list:h}));if(g.getSidebar().data(d)){g.addApplicationsCog()}g.bindAnalyticsHandlers(g.getSidebar(),h)})};this.updateProjectShortcuts=function(k,i,j){var l=k[0].shortcuts,h=i[0].shortcuts;g.getSidebar().find(".app-switcher-shortcuts").html(navlinks.templates.appswitcher.linkSection({title:"Shortcuts",list:l.concat(h)}));if(g.getSidebar().data(d)){g.addProjectShortcutsCog(j.key,j.entityType)}g.populateProjectHeader(j.name,j.avatarUrl);g.bindAnalyticsHandlers(g.getSidebar(),data)};this.entityMappings={"confluence.space":"com.atlassian.applinks.api.application.confluence.ConfluenceSpaceEntityType","jira.project":"com.atlassian.applinks.api.application.jira.JiraProjectEntityType","bamboo.project":"com.atlassian.applinks.api.application.bamboo.BambooProjectEntityType","stash.project":"com.atlassian.applinks.api.application.stash.StashProjectEntityType"};this.generateEntityLinksUrl=function(h,i){if(i===g.entityMappings["confluence.space"]){return AJS.contextPath()+"/spaces/listentitylinks.action?typeId="+i+"&key="+h}else{return AJS.contextPath()+"/plugins/servlet/applinks/listEntityLinks/"+i+"/"+h}};this.showAppSwitcherError=function(){e(function(){var h=g.getSidebar();h.find(".app-switcher-applications .app-switcher-loading").replaceWith(navlinks.templates.appswitcher.error());h.off(".appswitcher").on("click.appswitcher",".app-switcher-retry",e.proxy(g.retryLoading,g))})};this.showProjectShortcutsError=function(){e(function(){var h=g.getSidebar();h.find(".app-switcher-shortcuts .app-switcher-loading").replaceWith(navlinks.templates.appswitcher.error());h.off(".appswitcher").on("click.appswitcher",".app-switcher-retry",e.proxy(g.retryLoading,g))})};this.retryLoading=function(h){this.getSidebar().html(navlinks.templates.appswitcher.sidebarContents());this.getLinks();this.getProjectData();h&&h.stopPropagation()};this.bindAnalyticsHandlers=function(h,i){};this.getLinks();e(this.getProjectData);this.toggleSidebar=function(j){j.preventDefault();var k=g.getSidebar(),i=e("body"),h=e(window.document);if(!i.hasClass("app-switcher-open")){var m=e("#header");k.css("left",-k.width());k.parent("body").length||k.appendTo("body");b({data:k});k.animate({left:0},300);function l(n){var p=n.target&&e(n.target),o=n.keyCode;if(n.originalEvent===j.originalEvent){return}if(p&&!o&&!(p.closest(k).length||p.closest(m).length)&&j.which==1&&!(n.shiftKey||n.ctrlKey||n.metaKey)){g.toggleSidebar()}else{if(o===27){g.toggleSidebar()}}}h.on("click.appSwitcher",l);h.on("keydown.appSwitcher",l);h.on("scroll.appSwitcher",k,b)}else{h.off(".appSwitcher")}i.toggleClass("app-switcher-open")};e("#header").on("click",".app-switcher-trigger",this.toggleSidebar)};function b(h){var f=e(document).scrollTop(),i=e("#header"),g=(i.height()+i.offset().top)-f;if(g>=0){h.data.css({top:g,position:"fixed"})}else{h.data.css({top:0,left:0,position:"fixed"})}}e(function(){if(e(c).data("is-sidebar")===true){new a.SideBar({sidebarContents:c})}})}(jQuery,window.NL=(window.NL||{})));
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:rotp-menu', location = 'appswitcher/appswitcher_old.js' */
var APPSWITCHER_TRIGGER_CLICK="appswitcher.trigger.click";var APPSWITCHER_DROPDOWN_SHOW="appswitcher.dropdown.show";var APPSWITCHER_DROPDOWN_DISPLAY_ERROR="appswitcher.dropdown.display.error";var APPSWITCHER_APP_LINK_CLICK="appswitcher.app.link.click";var APPSWITCHER_CONFIGURE_LINK_CLICK="appswitcher.configure.link.click";(function(c,f){var b="isAppSuggestionAvailable";var d="isSiteAdminUser";var e="isUserAdmin";var a="#app-switcher";f.AppSwitcher=function(j){var l=AJS.contextPath()+"/plugins/servlet/customize-application-navigator";var k="unified.usermanagement";var m=this;this.$dropdown=null;j=c.extend({dropdownContents:null},j);this.getLinks=function(){return c.ajax({url:AJS.contextPath()+"/rest/menu/latest/appswitcher",cache:false,dataType:"json"}).done(this.updateDropdown).fail(this.showError)};this.getDropdown=function(){if(!this.$dropdown){this.$dropdown=c(j.dropdownContents);this.envData=this.$dropdown.data("environment")}return this.$dropdown};this.updateDropdown=function(n){c(function(){m.getDropdown().html(navlinks.templates.appswitcher_old.applications({apps:n,showAdminLink:m.envData[e],adminLink:l}));m.bindAnalyticsHandlers();if(m.envData[b]===true){m.handleSuggestionApps(n)}})};this.bindAnalyticsHandlers=function(){c(".app-switcher-trigger").on("click",function(){AJS.trigger("analyticsEvent",{name:APPSWITCHER_TRIGGER_CLICK})});c("#app-switcher").on("aui-dropdown2-show",function(){AJS.trigger("analyticsEvent",{name:APPSWITCHER_DROPDOWN_SHOW})});c("#app-switcher .nav-link").on("click",function(){var p="custom";var q=c(this).find("a");var o=q.attr("href");var n=window.location.hostname;if(o&&o.indexOf("bitbucket.org")>-1){p="bitbucket-cloud"}else{if(o.indexOf(n+"/wiki")>-1){p="confluence"}else{if(o.indexOf(n+"/build")>-1){p="bamboo"}else{if(o.indexOf(n)>-1){p="jira"}}}}AJS.trigger("analyticsEvent",{name:APPSWITCHER_APP_LINK_CLICK,data:{product:p}})});c(".nav-link-edit-wrapper").on("click",function(){AJS.trigger("analyticsEvent",{name:APPSWITCHER_CONFIGURE_LINK_CLICK})})};this.isBillingSystemEnabled=function(){return(this.envData[d]===true)&&AJS.DarkFeatures.isEnabled(k)};this.handleSuggestionApps=function(q){var r=_.map(q,function(s){return s.applicationType.toLowerCase()});var o=c("<div id='app-switcher-suggestion-apps' class='aui-dropdown2-section'/>");o.html(navlinks.templates.appswitcher_old.suggestionApps);var p=o.find(".suggestion-apps");var n=false;_.each(g,function(s){if(!_.contains(r,s.appName)){n=true;p.append(navlinks.templates.appswitcher_old.suggestionApp({suggestionApp:s,isBillingSystemEnabled:m.isBillingSystemEnabled()}))}});if(!n){return}c("#app-switcher").append(o);c(".app-discovery-suggestion-app").click(function(){var t=c(this).find("a");var s;if(m.envData[d]===true){s="appswitcher.discovery.siteadmin.select.inproduct."}else{s="appswitcher.discovery.user.select."}s=s+t.attr("id").toLowerCase();AJS.trigger("analytics",{name:s})});c(".app-discovery-suggestion-app").hover(function(){c(this).find("a").removeClass("active").removeClass("aui-dropdown2-active")});c(".app-discovery-cancel-button").click(function(){AJS.trigger("analytics",{name:"appswitcher.discovery.nothanks.button.click"});i(h,"true");o.remove()})};this.showError=function(){c(function(){AJS.trigger("analyticsEvent",{name:APPSWITCHER_DROPDOWN_DISPLAY_ERROR});m.getDropdown().html(navlinks.templates.appswitcher_old.error()).off(".appswitcher").on("click.appswitcher",".app-switcher-retry",c.proxy(m.retryLoading,m))})};this.retryLoading=function(n){this.getDropdown().html(navlinks.templates.appswitcher_old.loading());this.getLinks();n&&n.stopPropagation()};this.getLinks()};var h="key-no-thanks";var g=[{appName:"jira",appDesc:"Issue & Project Tracking Software",discoveryUrl:"https://www.atlassian.com/software/jira",billingSystemDiscoveryUrl:"/admin/billing/addapplication"},{appName:"confluence",appDesc:"Collaboration and content sharing",discoveryUrl:"https://www.atlassian.com/software/confluence",billingSystemDiscoveryUrl:"/admin/billing/addapplication?product=confluence.ondemand"},{appName:"bamboo",appDesc:"Continuous integration",discoveryUrl:"https://www.atlassian.com/software/bamboo",billingSystemDiscoveryUrl:"/admin/billing/addapplication?product=bamboo.ondemand"}];var i=function(j,k){c.ajax({url:AJS.contextPath()+"/rest/menu/latest/userdata/",type:"PUT",contentType:"application/json",data:JSON.stringify({key:j,value:k})})};c(function(){if(c(a).data("is-switcher")===true){new f.AppSwitcher({dropdownContents:a})}})}(jQuery,window.NL=(window.NL||{})));
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.plugins.atlassian-nav-links-plugin:rotp-menu', location = 'appswitcher/appswitcher_old.soy' */
// This file was automatically generated from appswitcher_old.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace navlinks.templates.appswitcher_old.
 */

if (typeof navlinks == 'undefined') { var navlinks = {}; }
if (typeof navlinks.templates == 'undefined') { navlinks.templates = {}; }
if (typeof navlinks.templates.appswitcher_old == 'undefined') { navlinks.templates.appswitcher_old = {}; }


navlinks.templates.appswitcher_old.applications = function(opt_data, opt_ignored) {
  return '' + navlinks.templates.appswitcher_old.applicationsSection({list: opt_data.apps, listClass: 'nav-links', showDescription: opt_data.showDescription}) + ((opt_data.custom) ? navlinks.templates.appswitcher_old.applicationsSection({list: opt_data.custom, showDescription: opt_data.showDescription}) : '') + ((opt_data.showAdminLink) ? navlinks.templates.appswitcher_old.adminSection(opt_data) : '');
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.applications.soyTemplateName = 'navlinks.templates.appswitcher_old.applications';
}


navlinks.templates.appswitcher_old.applicationsSection = function(opt_data, opt_ignored) {
  var output = '';
  if (opt_data.list.length > 0) {
    var param19 = '<ul' + ((opt_data.listClass) ? ' class="' + soy.$$escapeHtml(opt_data.listClass) + '"' : '') + '>';
    var linkList27 = opt_data.list;
    var linkListLen27 = linkList27.length;
    for (var linkIndex27 = 0; linkIndex27 < linkListLen27; linkIndex27++) {
      var linkData27 = linkList27[linkIndex27];
      param19 += navlinks.templates.appswitcher_old.applicationsItem(soy.$$augmentMap(linkData27, {showDescription: opt_data.showDescription}));
    }
    param19 += '</ul>';
    output += aui.dropdown2.section({content: param19});
  }
  return output;
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.applicationsSection.soyTemplateName = 'navlinks.templates.appswitcher_old.applicationsSection';
}


navlinks.templates.appswitcher_old.applicationsItem = function(opt_data, opt_ignored) {
  return '<li class="nav-link' + ((opt_data.self) ? ' nav-link-local' : '') + '"><a href="' + soy.$$escapeHtml(opt_data.link) + '" class="aui-dropdown2-radio ' + ((opt_data.self) ? 'aui-dropdown2-checked' : '') + '" title="' + soy.$$escapeHtml(opt_data.link) + '"><span class="nav-link-label">' + soy.$$escapeHtml(opt_data.label) + '</span>' + ((opt_data.showDescription && opt_data.description) ? '<span class="nav-link-description">' + soy.$$escapeHtml(opt_data.description) + '</span>' : '') + '</a></li>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.applicationsItem.soyTemplateName = 'navlinks.templates.appswitcher_old.applicationsItem';
}


navlinks.templates.appswitcher_old.adminSection = function(opt_data, opt_ignored) {
  return '' + aui.dropdown2.section({content: '<ul class="nav-links"><li><a class="nav-link-edit-wrapper" href="' + soy.$$escapeHtml(opt_data.adminLink) + '"><span class="nav-link-edit">' + soy.$$filterNoAutoescape('Configure...') + '</span></a></li></ul>'});
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.adminSection.soyTemplateName = 'navlinks.templates.appswitcher_old.adminSection';
}


navlinks.templates.appswitcher_old.error = function(opt_data, opt_ignored) {
  return '<div class="app-switcher-error">' + soy.$$filterNoAutoescape('Something went wrong, please \x3cspan class\x3d\x22app-switcher-retry\x22\x3etry again\x3c/span\x3e.') + '</div>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.error.soyTemplateName = 'navlinks.templates.appswitcher_old.error';
}


navlinks.templates.appswitcher_old.loading = function(opt_data, opt_ignored) {
  return '<div class="app-switcher-loading">' + soy.$$filterNoAutoescape('Loading...') + '</div>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.loading.soyTemplateName = 'navlinks.templates.appswitcher_old.loading';
}


navlinks.templates.appswitcher_old.trigger = function(opt_data, opt_ignored) {
  return '<span class="aui-icon aui-icon-small aui-iconfont-appswitcher">' + soy.$$escapeHtml('Linked Applications') + '</span>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.trigger.soyTemplateName = 'navlinks.templates.appswitcher_old.trigger';
}


navlinks.templates.appswitcher_old.switcher = function(opt_data, opt_ignored) {
  var output = '';
  if (true) {
    var loadingContent__soy81 = '' + navlinks.templates.appswitcher_old.loading(null);
    var triggerContent__soy83 = '' + navlinks.templates.appswitcher_old.trigger(null);
    output += aui.dropdown2.dropdown2({menu: {id: 'app-switcher', content: loadingContent__soy81, extraClasses: 'aui-style-default', extraAttributes: {'data-environment': {}, 'data-is-switcher': 'true'}}, trigger: {tagName: 'button', showIcon: false, content: triggerContent__soy83, extraClasses: 'app-switcher-trigger aui-dropdown2-trigger-arrowless', extraAttributes: {href: '#app-switcher'}}});
  }
  return output;
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.switcher.soyTemplateName = 'navlinks.templates.appswitcher_old.switcher';
}


navlinks.templates.appswitcher_old.suggestionApp = function(opt_data, opt_ignored) {
  var output = '';
  var href__soy89 = opt_data.isBillingSystemEnabled == true ? opt_data.suggestionApp.billingSystemDiscoveryUrl : opt_data.suggestionApp.discoveryUrl;
  output += '<li class="app-discovery-suggestion-app"><a id="' + soy.$$escapeHtml(opt_data.suggestionApp.appName) + '" href="' + soy.$$escapeHtml(href__soy89) + '" class="app-discovery-link aui-icon-container app-discovery-' + soy.$$escapeHtml(opt_data.suggestionApp.appName) + '-product-icon" title="' + soy.$$escapeHtml(href__soy89) + '" target="_blank"></a><div class="app-discovery-small">' + soy.$$escapeHtml(opt_data.suggestionApp.appDesc) + '</div></li>';
  return output;
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.suggestionApp.soyTemplateName = 'navlinks.templates.appswitcher_old.suggestionApp';
}


navlinks.templates.appswitcher_old.suggestionApps = function(opt_data, opt_ignored) {
  return '<ul class=\'nav-links suggestion-apps\'><li><span class=\'app-discovery-suggest-title nav-link-label\'><h6>' + soy.$$escapeHtml('Try other Atlassian apps') + '</h6></span></li></ul><div class=\'buttons-container app-discovery-suggest-apps-buttons\'><div class=\'buttons\'><button class=\'aui-button aui-button-link app-discovery-cancel-button\' name=\'cancel\' accesskey=\'c\' href=\'#\'>' + soy.$$escapeHtml('Don\x27t show this again') + '</button></div></div>';
};
if (goog.DEBUG) {
  navlinks.templates.appswitcher_old.suggestionApps.soyTemplateName = 'navlinks.templates.appswitcher_old.suggestionApps';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:common-header-resources', location = '/includes/js/header-user-menu.js' */
define("confluence/header-user-menu",["jquery","confluence/meta","underscore"],function(a,c,b){function d(){var b=c.get("current-user-avatar-uri-reference");a("#user-menu-link").find(".aui-avatar-inner img").attr("src",b)}return function(){b.defer(d)}});require("confluence/module-exporter").safeRequire("confluence/header-user-menu",function(a){require("ajs").toInit(a)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:general-analytics-bindings', location = '/includes/js/analytics-bindings.js' */
define("confluence/analytics-bindings",["jquery","confluence/analytics-support"],function(b,d){return function(){function e(a,c){b(a).on("click",".view-historical-version-trigger",function(){d.publish("confluence.page.view-historical.from-"+(c||"unknown"))})}function f(a,c){b(a).on("click",".restore-historical-version-trigger",function(){d.publish("confluence.page.restore-historical.from-"+(c||"unknown"))})}function g(a,c){b("#header .aui-header-secondary "+a).on("click",function(){var a=b(this).hasClass("aui-dropdown2-active")?
"expanded":"collapsed";d.publish("confluence.header.dropdown."+c,{state:a})})}e("#page-history-warning","navigation");e("#page-history-container","history");e(".diff-menu","diff");f("#page-history-warning","navigation");f("#page-history-container","history");g("#admin-menu-link","administration");g("#user-menu-link","profile")}});require("confluence/module-exporter").safeRequire("confluence/analytics-bindings",function(b){require("ajs").toInit(b)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickreload:quick-reload-read-only-mode-bootstrap', location = 'main/quick-reload-read-only-mode-bootstrap.js' */
require(["ajs","confluence/dark-features","confluence-quick-reload/main/quick-reload-manager","confluence-quick-reload/handlers/quick-reload-read-only-mode"],function(b,c,a,d){b.toInit(function(){c.isEnabled("quickreload.disabled")||(a.addHandler(d),a.enable())})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.mywork.mywork-confluence-host-plugin:mw-header-anchor', location = 'templates/anchor.soy' */
// This file was automatically generated from anchor.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace MyWork.Templates.Anchor.
 */

if (typeof MyWork == 'undefined') { var MyWork = {}; }
if (typeof MyWork.Templates == 'undefined') { MyWork.Templates = {}; }
if (typeof MyWork.Templates.Anchor == 'undefined') { MyWork.Templates.Anchor = {}; }


MyWork.Templates.Anchor.tasksFeatureDiscovery = function(opt_data, opt_ignored) {
  return '<p>' + soy.$$escapeHtml('See the tasks assigned to you, or created by you, in the Tasks tab of your profile.') + '</p><ul class="mw-tasks-discovery-controls"><li><a id="mw-tasks-discovery-view" href="' + soy.$$escapeHtml(opt_data.tasksUrl) + '" class="aui-button aui-style"><p>' + soy.$$escapeHtml('View Tasks') + '</p></a></li><li><a id="mw-tasks-discovery-dismiss" href="#">' + soy.$$escapeHtml('Dismiss') + '</a></li></ul>';
};
if (goog.DEBUG) {
  MyWork.Templates.Anchor.tasksFeatureDiscovery.soyTemplateName = 'MyWork.Templates.Anchor.tasksFeatureDiscovery';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.mywork.mywork-confluence-host-plugin:mw-header-anchor', location = 'js/miniview-anchor.js' */
var MW=MW||{$:jQuery};MW.MV={};
AJS.toInit(function(){if(!AJS.Meta||AJS.Meta.get("remote-user")){MW.MV.AnchorManager=function(){function y(){MW.$(document).keydown(function(a){AJS.InlineDialog.current&&27==a.which&&!MW.$(a.target).is(":input")&&AJS.InlineDialog.current.hide()})}function z(){MW.$("#header-menu-bar").find(".ajs-drop-down").each(function(){this.hide()})}function t(a){return MW.$("body",a.document).find("a[href]:visible, button:not([disabled]), input, textarea, select, details,[tabindex]:not([tabindex\x3d'-1']):visible")}function A(a,
e){(new IntersectionObserver(function(f){f.forEach(function(n){0<n.intersectionRatio&&e()})})).observe(a)}function B(a,e){function f(b){if("escKey"===b.data)l();else if("getParentConfig"===b.data&&b.origin===location.protocol+"//"+location.host){b=JSON.stringify({parentConfig:{parentUrl:location.href,preload:MW.Dialog.preload,unread:p}});var c=MW.$("#"+a+"-miniview-iframe")[0].contentWindow;c.postMessage(b,"*");c.document.addEventListener("keydown",n(c));MW.Dialog.preload&&(MW.Dialog.preload=!1)}}
function n(b){return function(c){if("Tab"==c.key){var d=t(b),g=d[0],q=d[d.length-1];d.length?c.shiftKey&&b.document.activeElement===g?l():b.document.activeElement===q&&l():l()}}}function l(){MW.$("#notifications-anchor").focus();u.hide()}var u;window.addEventListener?window.addEventListener("message",f,!1):window.attachEvent("onmessage",f);MW.Dialog=AJS.InlineDialog(MW.$("#"+a+"-anchor"),a+"-miniview",function(b,c,d){0===MW.$(b).children().length?MW.$(b).append(MW.$('\x3ciframe id\x3d"'+a+'-miniview-iframe" src\x3d"'+
e+'" frameborder\x3d"0"\x3e\x3c/iframe\x3e')):(b=JSON.stringify({unread:p}),MW.$("#"+a+"-miniview-iframe")[0].contentWindow.postMessage(b,"*"));z();d()},{width:500,height:520,hideDelay:null,initCallback:function(){u=this;MW.$("#notifications-anchor").attr("aria-expanded","true");var b=MW.$("#"+a+"-miniview-iframe")[0].contentWindow;A(MW.$("#inline-dialog-notifications-miniview")[0],function(){var c=t(b)[0];c?c.focus():b.focus()})},hideCallback:function(){if(!0!==this.preload){var b=JSON.stringify({markAllRead:!0});
MW.$("#"+a+"-miniview-iframe")[0].contentWindow.postMessage(b,"*");MW.$("#notifications-anchor").attr("aria-expanded","false")}},noBind:!0});MW.Tasks=function(){var b,c=$("#user-menu-link"),d=$("#user-menu-link-content"),g=d.find("#view-mytasks-link"),q=function(h,k,r){k=MyWork.Templates.Anchor.tasksFeatureDiscovery({tasksUrl:AJS.contextPath()+"/plugins/inlinetasks/mytasks.action"});h.html(k);h.find("#mw-tasks-discovery-dismiss").click(function(){b.hide()});r()};return{closeAndDiscoverMyTasks:function(){MW.Dialog.hide();
d.is(":visible")||c.trigger("aui-button-invoke");var h=function(){b.hide()};d.one("aui-dropdown2-hide",h);b=AJS.InlineDialog(g,"my-tasks-discovery",q,{hideCallback:function(){b.unbind("click focusin mousedown",k);d.unbind("aui-dropdown2-hide",h);d.is(":visible")&&c.trigger("aui-button-invoke");MW.$("#inline-dialog-my-tasks-discovery").remove()},gravity:"w",useLiveEvents:!0,width:300,noBind:!0});d.find(".user-item.active").removeClass("active");g.addClass("active");g.focus();var k=function(r){r.stopPropagation()};
b.on("click focusin mousedown",k);b.show()}}}();MW.$("#"+a+"-anchor").click(function(b){b.preventDefault();MW.$("#"+a+"-miniview-iframe").is(":visible")?MW.Dialog.hide():MW.Dialog.show()});v&&MW.$("#"+a+"-anchor").click()}var w=contextPath+"/plugins/servlet/notifications-miniview",p=0,v=/[?&]show-miniview/.test(window.location.search),x=function(a){a=a.replace(/[\[]/,"\\[").replace(/[\]]/,"\\]");a=(new RegExp("[\\?\\#\x26]"+a+"\x3d([^\x26#]*)")).exec(window.location.search);if(null!==a)return decodeURIComponent(a[1].replace(/\+/g,
" "))}("show-miniview");x&&(w+="#notification/"+x);var C=function(a){var e=!1,f;return function(){if(e)return f;e=!0;return f=a.apply(this,arguments)}}(function(){MW.Dialog.getOptions().closeOthers=!1;MW.Dialog.preload=!0;MW.Dialog.show();MW.Dialog.hide();MW.Dialog.getOptions().closeOthers=!0});return{setupAnchors:function(){MW.$("#notifications-anchor").html('\x3cdiv class\x3d"badge-i aui-icon aui-icon-small aui-iconfont-notification"\x3e\x3c/div\x3e\x3cspan class\x3d"badge-w"\x3e\x3cspan class\x3d"badge" aria-hidden\x3d"true"\x3e\x3c/span\x3e\x3c/span\x3e').attr({title:"Open Notifications",
role:"button","aria-label":"Notifications","aria-expanded":"false"});B("notifications",w);y()},updateNotificationCount:function(a){var e=MW.$("#notifications-anchor"),f=e.find(".badge");e.find(".aui-icon");f.html(9>=a?a:"9+");p=a;0<a?(e.addClass("unread").removeClass("read").attr("aria-label",AJS.format("Notifications: {0} new",a)),e.is(":visible")&&!v&&C()):e.addClass("read").removeClass("unread").attr("aria-label","Notifications")}}}();
MW.MV.AnchorManager.setupAnchors();var m=new MW.AnchorUtil(MW.$,contextPath,MW.MV.AnchorManager.updateNotificationCount);m.setupAnchors();MW.$("#notifications-anchor").click(function(){AJS.trigger("analytics",{name:"mywork.host.button.clicked.notifications",data:{}});MW.MV.AnchorManager.updateNotificationCount(0)});document.addEventListener("visibilitychange",function(){document.hidden?m.stopRequests(!0):m.startRequests()},!1);MW.$("body").click(function(){m.startRequests()})}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.mywork.mywork-confluence-host-plugin:mw-header-anchor', location = 'js/util/anchor-util.js' */
MW.AnchorUtil=function(t,n,p){function e(){3E5>(new Date).getTime()-f&&a||l();f=(new Date).getTime()}function l(){a&&clearTimeout(a);a=setTimeout(function(){c()},m=q(m))}function q(b){return Math.min(3E5>(new Date).getTime()-f?d:1.25*b,g)}function c(b){MW.$.getJSON(r+(b?"?pageid\x3d"+b:""),function(h){var k=1E3*h.timeout;g=1E3*h.maxTimeout||g;k&&k!=d&&(d=k,e());p(h.count)});l()}var d=3E4,a,r=n+"/rest/mywork/latest/status/notification/count",f=(new Date).getTime(),g=3E5,m=0;return{setupAnchors:function(){AJS&&
AJS.Meta&&AJS.Meta.get&&("page"===AJS.Meta.get("content-type")||"blogpost"===AJS.Meta.get("content-type"))?c(AJS.Meta.get("page-id")):c();e()},startRequests:e,stopRequests:function(b){window.clearInterval(a);a=void 0;!0===b&&(d=3E4)},updateAnchors:c}};
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-search-ui-plugin:search-ui-client-templates', location = 'soy/search-ui-client-templates.soy' */
// This file was automatically generated from search-ui-client-templates.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.SearchUI.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.SearchUI == 'undefined') { Confluence.Templates.SearchUI = {}; }


Confluence.Templates.SearchUI.helpUrl = function(opt_data, opt_ignored) {
  return '' + soy.$$escapeHtml("https:\/\/docs.atlassian.com\/confluence\/docs-713\/Search");
};
if (goog.DEBUG) {
  Confluence.Templates.SearchUI.helpUrl.soyTemplateName = 'Confluence.Templates.SearchUI.helpUrl';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-search-ui-plugin:confluence-search-ui-plugin-resources', location = '/js/confluence-search-ui-analytics-support.js' */
'use strict';

/**
 * Utility functions for publishing new search ui analytics
 * @module confluence/searchui/analytics-support
 */
define('confluence/search-ui/analytics-support', ['window', 'document', 'confluence/api/event', 'confluence/api/url'], function (window, document, eventApi, url) {
    "use strict";

    var SEARCH_UI_LEAVE_CONTENT_FROM_SEARCH_RESULT = 'confluence-search-ui-leave-content-from-new-search';
    var SEARCH_UI_EDIT_CONTENT_FROM_SEARCH_RESULT = 'confluence-search-ui-edit-content-from-new-search';
    /**
     * trigger analytic
     * @since 6.13.0
     * @public
     * @param name the analytic name
     * @param data the analytic data
     */
    function triggerAnalytic(name, data) {
        data = data || {};
        eventApi.trigger('analytics', { name: name, data: data });
    }

    /**
     * @param urlString
     * @returns {*} return array of searchId
     */
    function extractAnalyticsData(urlString) {
        var params = url.parse(urlString).queryParams;
        return params && params.searchId ? params.searchId : [];
    }

    /**
     * trigger period analytic event when beforeunload event happen
     * @param searchId searchId
     * @param period the period in millisecond
     */
    function triggerPeriodEventViewContentFromSearch(searchId, period) {
        triggerAnalytic(SEARCH_UI_LEAVE_CONTENT_FROM_SEARCH_RESULT, { searchId: searchId, period: period });
    }

    /**
     * trigger period analytic event when enter editor init.rte
     * @param searchId searchId
     * @param period the period in millisecond
     */
    function triggerPeriodEvenEditContentFromSearch(searchId, period) {
        triggerAnalytic(SEARCH_UI_EDIT_CONTENT_FROM_SEARCH_RESULT, { searchId: searchId, period: period });
    }

    /**
     * Remove
     * @param urlString
     * @returns {*}
     */
    function removeSearchIdUrl(urlString) {
        var parsedUrl = url.parse(urlString);
        delete parsedUrl.queryParams.searchId;
        return url.format(parsedUrl);
    }

    /**
     * Clean analytic parameter
     */
    function replaceStateAfterCleaningUpAnalyticsParameters() {
        var cleanUrl = removeSearchIdUrl(document.URL);
        if (document.URL !== cleanUrl) {
            window.history.replaceState(null, '', cleanUrl);
        }
    }

    /**
     * Init analytic for period
     * @since 6.13
     * @public
     */
    function initAnalytic() {
        var analyticsData = extractAnalyticsData(document.URL);
        var searchId = analyticsData[0];
        if (!searchId) {
            return;
        }

        var currentDate = new Date();
        window.addEventListener('beforeunload', function () {
            var currentTriggerTime = new Date();
            var period = currentTriggerTime.getTime() - currentDate.getTime();
            triggerPeriodEventViewContentFromSearch(searchId, period);
        });
        eventApi.bind('init.rte', function () {
            var currentTriggerTime = new Date();
            var period = currentTriggerTime.getTime() - currentDate.getTime();
            triggerPeriodEvenEditContentFromSearch(searchId, period);
        });
        // Check if browser supports HTML5 replaceState()
        if (window.history && window.history.replaceState) {
            replaceStateAfterCleaningUpAnalyticsParameters();
        }
    }

    return {
        init: initAnalytic
    };
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-search-ui-plugin:confluence-search-ui-plugin-resources', location = '/js/confluence-search-ui-plugin.js' */
require(['ajs', 'wrm', 'document', 'confluence/api/logger', 'confluence/meta', 'confluence/api/constants', 'confluence/search-ui/analytics-support'],
    function (ajs, wrm, document, logger, meta, constants, analyticsSupport) {
        const events = ['click', 'focus'];
        var searchUI;
        var searchInput;
        var spinner;
        var tooltip;
        var isTooltipShown = false;
        var isLoading = false;

        function toggleIcon() {
            searchUI.classList.toggle("hide-icon");
        }

        function startLoading() {
            isLoading = true;
            toggleIcon();
            spinner.style.display = 'block';
        }

        function stopLoading() {
            isLoading = false;
            toggleIcon();
            spinner.style.display = 'none';
        }

        function unregisterHandlers() {
            events.forEach(function (event) {
                searchInput.removeEventListener(event, lazyLoad);
            });
        }

        function showTooltip() {
            if (!isTooltipShown) {
                tooltip.style.display = 'block';
                isTooltipShown = true;
            }
        }

        function hideTooltip() {
            if (isTooltipShown) {
                tooltip.style.display = 'none';
                isTooltipShown = false;
            }
        }

        function handleClickAndFocusEvent(event) {
            showTooltip();
            event.stopPropagation();
        }

        function handleInputEvent(event) {
            if (event.target.value) {
                showTooltip();
            } else {
                hideTooltip();
            }
        }

        function handleKeyDownEvent(event) {
            const ENTER_KEY_CODE = 13;
            if (event.keyCode === ENTER_KEY_CODE) {
                const searchTerm = event.target.value;
                const cql = searchTerm ? '?cql=&queryString=' + encodeURIComponent(searchTerm) : '';
                window.location.href = constants.CONTEXT_PATH + '/dosearchsite.action' + cql;
            }
        }

        function handleDocumentClick(event) {
            if (searchUI.contains(event.target)) {
                showTooltip();
            } else {
                hideTooltip();
            }
        }

        function lazyLoad() {
            if (isLoading) return;

            startLoading();
            wrm
                .require(['wrc!confluence-search-ui-plugin-main'])
                .done(function () {
                    require(['confluence-search-ui-plugin-main'], function (SearchUI) {
                        SearchUI.init();
                        unregisterHandlers();
                        stopLoading();
                    });
                })
                .fail(function () {
                    stopLoading();
                    logger.error('fail to load confluence-search-ui-plugin-main');
                });
        }

        ajs.toInit(function () {
            searchUI = document.getElementById('search-ui');
            // login screen doesn't have navigation search
            if (!searchUI) {
                logger.log('no search-ui element found');
                return;
            }

            analyticsSupport.init();
            searchInput = searchUI.querySelector('#quick-search-query');
            spinner = searchUI.querySelector('aui-spinner');
            tooltip = searchUI.querySelector('[role=alert]');

            if (!meta.get('global-settings-quick-search-enabled')) {
                searchInput.addEventListener('keydown', handleKeyDownEvent);
                searchInput.addEventListener('input', handleInputEvent);

                events.forEach(function (event) {
                    searchInput.addEventListener(event, handleClickAndFocusEvent);
                });

                document.addEventListener('click', handleDocumentClick);
            } else {
                events.forEach(function (event) {
                    searchInput.addEventListener(event, lazyLoad);
                });
            }
        });
    });

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.dialog-wizard:dialog-wizard-resources', location = '/soy/dialog-wizard.soy' */
// This file was automatically generated from dialog-wizard.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.DialogWizard.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.DialogWizard == 'undefined') { Confluence.Templates.DialogWizard = {}; }


Confluence.Templates.DialogWizard.pageContainer = function(opt_data, opt_ignored) {
  return '<div class="dialog-wizard-page-wrapper"><div class="dialog-wizard-page-main"></div><div class="dialog-wizard-page-description">' + ((opt_data.descriptionHeaderLink && opt_data.descriptionHeader) ? '<h3><a href=\'' + soy.$$escapeHtml(opt_data.descriptionHeaderLink) + '\' target=\'_blank\'>' + soy.$$escapeHtml(opt_data.descriptionHeader) + '</a></h3>' : (opt_data.descriptionHeader) ? '<h3>' + soy.$$escapeHtml(opt_data.descriptionHeader) + '</h3>' : '') + '<p>' + soy.$$escapeHtml(opt_data.descriptionContent) + '</p></div></div>';
};
if (goog.DEBUG) {
  Confluence.Templates.DialogWizard.pageContainer.soyTemplateName = 'Confluence.Templates.DialogWizard.pageContainer';
}


Confluence.Templates.DialogWizard.waitIcon = function(opt_data, opt_ignored) {
  return '<img class="wait-icon" src="' + soy.$$escapeHtml("/wiki") + '/images/icons/wait.gif">';
};
if (goog.DEBUG) {
  Confluence.Templates.DialogWizard.waitIcon.soyTemplateName = 'Confluence.Templates.DialogWizard.waitIcon';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.dialog-wizard:dialog-wizard-resources', location = '/js/dialog-wizard.js' */
(function($){var FormStateControl=require("confluence/form-state-control");function findNextPageId(pageId,pages){var thisPageIdIndex=-1;_.each(pages,function(element,index){if(element.id==pageId){thisPageIdIndex=index;return}});return pages[thisPageIdIndex+1].id}Confluence.DialogWizard=function(dialog,finalAction){function newPage(config,pageId,soyRenderContext,wizardData,wizard){var wizardPage=_.find(config.wizard.pages,function(page){return page.id==pageId});wizard.trigger("pre-render."+pageId,{soyRenderContext:soyRenderContext,wizardData:wizardData});try{var soyTemplateFunction=eval(wizardPage.templateKey);var $soyRender=$(soyTemplateFunction(soyRenderContext))}catch(e){throw Error("wizard points to a non-existent Soy template '"+wizardPage.templateKey+"'. Check your web-resources or server logs.")}$soyRender.find("a, area, button, input, object, select, textarea").attr("tabindex","10");var $panelContent;if(wizardPage.descriptionContent){$panelContent=$(Confluence.Templates.DialogWizard.pageContainer({descriptionHeaderLink:wizardPage.descriptionHeaderLink,descriptionHeader:wizardPage.descriptionHeader,descriptionContent:wizardPage.descriptionContent}));$panelContent.addClass("with-description").find(".dialog-wizard-page-main").append($soyRender)}else{$panelContent=$soyRender}var dialogPageId=pageId;if(dialog.id=="create-dialog"){dialogPageId="create-dialog-"+pageId}var page=dialog.addPage(dialogPageId).page[dialog.curpage];page.addHeader(wizardPage.title).addPanel("SinglePanel",$panelContent,"singlePanel");page.element.find("form").submit(function(){return false});if(wizardPage.descriptionContent){page.element.find(".dialog-panel-body").css({padding:0})}Confluence.Binder.autocompleteMultiUser($soyRender);Confluence.Binder.placeholder($soyRender);function nextCallback(ev){$soyRender.find(".placeholded").val("");var pageData={};var formArray=$soyRender.parent().find("form").serializeArray();_.each(formArray,function(item){pageData[item.name]=item.value});var e=$.Event("submit."+pageId);var state={$container:$soyRender,wizardData:wizardData,pageData:pageData};var validationDeferred=$.Deferred();validationDeferred.done(function(){wizardData.pages[pageId]=pageData;var nextPageId;if(state.nextPageId){nextPageId=state.nextPageId}else{nextPageId=!wizardPage.last&&findNextPageId(pageId,config.wizard.pages)}if(!state.nextPageId&&wizardPage.last){doFinalAction(ev,state,wizardData,finalAction,wizard);FormStateControl.disableElement(dialog.popup.element.find(":input,a").filter(":visible"));buttons.prepend(Confluence.Templates.DialogWizard.waitIcon())}else{newPage(config,nextPageId,soyRenderContext,wizardData,wizard)}});validationDeferred.fail(function(){AJS.log("dialog aborted by on-submit callback on page: "+pageId)});state.validationDeferred=validationDeferred;wizard.trigger(e,state);if(state.async){return}if(e.isDefaultPrevented()){validationDeferred.reject();return}validationDeferred.resolve()}var buttons=dialog.addFullButtonPanel(page,nextCallback,null,wizardPage.last);buttons.find(".button-panel-back").click(function(){delete wizardData.pages[pageId]});buttons.find(".aui-button-primary").attr("tabindex","10");$soyRender.find("input, select, textarea").eq(0).focus();wizard.trigger("post-render."+pageId,{$container:$soyRender,wizardData:wizardData})}function doFinalAction(ev,state,wizardData,finalAction,wizard){if(state.finalUrl){if(ev.type==="click"&&ev.which===2){var windowOpened=window.open(state.finalUrl);windowOpened.opener=null}else{window.location=state.finalUrl}}else{_.each(wizardData.pages,function(pageData){_.extend(wizardData,pageData)});delete wizardData.pages;finalAction(ev,wizardData,null,wizard)}return{success:function(callback){callback()}}}return{newPage:newPage,doFinalAction:doFinalAction}}})(AJS.$);
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-business-blueprints:common-template-resources', location = 'com/atlassian/confluence/plugins/blueprint/common/soy/common-templates.soy' */
// This file was automatically generated from common-templates.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Blueprints.Common.Index.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Blueprints == 'undefined') { Confluence.Blueprints = {}; }
if (typeof Confluence.Blueprints.Common == 'undefined') { Confluence.Blueprints.Common = {}; }
if (typeof Confluence.Blueprints.Common.Index == 'undefined') { Confluence.Blueprints.Common.Index = {}; }


Confluence.Blueprints.Common.Index.detailsSummaryMacro = function(opt_data, opt_ignored) {
  return '<ac:macro ac:name="detailssummary"><ac:parameter ac:name="label">' + soy.$$escapeHtml(opt_data.label) + '</ac:parameter><ac:parameter ac:name="spaces">' + soy.$$escapeHtml(opt_data.spaces) + '</ac:parameter><ac:parameter ac:name="firstcolumn">' + soy.$$escapeHtml(opt_data.firstcolumn) + '</ac:parameter><ac:parameter ac:name="headings">' + soy.$$escapeHtml(opt_data.headings) + '</ac:parameter><ac:parameter ac:name="blankTitle">' + soy.$$escapeHtml(opt_data.blankTitle) + '</ac:parameter><ac:parameter ac:name="blankDescription">' + soy.$$escapeHtml(opt_data.blankDescription) + '</ac:parameter><ac:parameter ac:name="contentBlueprintId">' + soy.$$escapeHtml(opt_data.contentBlueprintId) + '</ac:parameter><ac:parameter ac:name="blueprintModuleCompleteKey">' + soy.$$escapeHtml(opt_data.blueprintModuleCompleteKey) + '</ac:parameter><ac:parameter ac:name="createButtonLabel">' + soy.$$escapeHtml(opt_data.createButtonLabel) + '</ac:parameter></ac:macro>';
};
if (goog.DEBUG) {
  Confluence.Blueprints.Common.Index.detailsSummaryMacro.soyTemplateName = 'Confluence.Blueprints.Common.Index.detailsSummaryMacro';
}


Confluence.Blueprints.Common.Index.createFromTemplateMacro = function(opt_data, opt_ignored) {
  return '<ac:macro ac:name="create-from-template"><ac:parameter ac:name="blueprintModuleCompleteKey">' + soy.$$escapeHtml(opt_data.moduleKey) + '</ac:parameter><ac:parameter ac:name="buttonLabel">' + soy.$$escapeHtml(opt_data.buttonLabel) + '</ac:parameter><ac:parameter ac:name="spaceKey">' + soy.$$escapeHtml(opt_data.spaceKey) + '</ac:parameter><ac:parameter ac:name="templateName">' + soy.$$escapeHtml(opt_data.templateName) + '</ac:parameter></ac:macro>';
};
if (goog.DEBUG) {
  Confluence.Blueprints.Common.Index.createFromTemplateMacro.soyTemplateName = 'Confluence.Blueprints.Common.Index.createFromTemplateMacro';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-business-blueprints:meeting-notes-resources', location = 'com/atlassian/confluence/plugins/meetingnotes/soy/how-to.soy' */
// This file was automatically generated from how-to.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Blueprints.Meeting.Notes.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Blueprints == 'undefined') { Confluence.Blueprints = {}; }
if (typeof Confluence.Blueprints.Meeting == 'undefined') { Confluence.Blueprints.Meeting = {}; }
if (typeof Confluence.Blueprints.Meeting.Notes == 'undefined') { Confluence.Blueprints.Meeting.Notes = {}; }


Confluence.Blueprints.Meeting.Notes.howTo = function(opt_data, opt_ignored) {
  return '<h1>' + soy.$$escapeHtml('With meeting notes you can...') + '</h1><ol class="howto-steps"><li class="howto-step"><div><h3>' + soy.$$escapeHtml('Crowd-source your agenda') + '</h3><p>' + soy.$$escapeHtml('Distribute an agenda and keep meetings focused.') + '</p></div></li><li class="howto-step"><div><h3>' + soy.$$escapeHtml('Capture meeting minutes') + '</h3><p>' + soy.$$escapeHtml('Take notes and make them available to everyone.') + '</p></div></li><li class="howto-step"><div><h3>' + soy.$$escapeHtml('Create and assign tasks') + '</h3><p>' + soy.$$escapeHtml('Assign action items for attendees to work on afterward.') + '</p></div></li></ol>';
};
if (goog.DEBUG) {
  Confluence.Blueprints.Meeting.Notes.howTo.soyTemplateName = 'Confluence.Blueprints.Meeting.Notes.howTo';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-business-blueprints:meeting-notes-resources', location = 'com/atlassian/confluence/plugins/meetingnotes/soy/user-mention.soy' */
// This file was automatically generated from user-mention.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Templates.Meeting.Notes.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Templates == 'undefined') { Confluence.Templates = {}; }
if (typeof Confluence.Templates.Meeting == 'undefined') { Confluence.Templates.Meeting = {}; }
if (typeof Confluence.Templates.Meeting.Notes == 'undefined') { Confluence.Templates.Meeting.Notes = {}; }


Confluence.Templates.Meeting.Notes.userMention = function(opt_data, opt_ignored) {
  opt_data = opt_data || {};
  return '' + ((opt_data.userkey) ? '<li><p><ac:link><ri:user ri:userkey="' + soy.$$escapeHtml(opt_data.userkey) + '" /></ac:link></p></li><li><p><ac:placeholder ac:type="mention">' + soy.$$escapeHtml('@mention a person to add them as an attendee and they will be notified.') + '</ac:placeholder></p></li>' : '<li><p><ac:placeholder ac:type="mention">' + soy.$$escapeHtml('@mention a person to add them as an attendee and they will be notified.') + '</ac:placeholder></p></li>');
};
if (goog.DEBUG) {
  Confluence.Templates.Meeting.Notes.userMention.soyTemplateName = 'Confluence.Templates.Meeting.Notes.userMention';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.troubleshooting.plugin-confluence:atst-common', location = 'js/ajs-amd.js' */
define('troubleshooting/ajs', [], function () {
    'use strict';

    return AJS;
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.troubleshooting.plugin-confluence:atst-healthcheck-sensors', location = 'js/sensors.js' */
require([
    'troubleshooting/sensors/page-protocols',
    'troubleshooting/ajs'
], function(
    protocolSensor,
    AJS
) {
    var sensors = [protocolSensor];

    AJS.toInit(function() {

        function hash(str) {
            var hash = 0;
            for (var i = 0; i < str.length; ++i) {
                hash = 31 * hash + str.charCodeAt(i);
                hash |= 0; // this reduces the number to 32bits and prevents Infinity
            }
            return hash;
        }

        //this handles cases when the user disabled access to local storage
        var localStorageWrapper = {
            getWithDefaultOnError: function(sensorName, defaultValue) {
                try {
                    return window.localStorage.getItem(sensorName) || "0";
                } catch (e) {
                    return defaultValue;
                }
            },
            setItemQuietly: function (key, value) {
                try {
                    window.localStorage.setItem(key, value);
                } catch (ignored) {
                }
            }
        };

        // Iterate through each sensor and ingest any data it has for us...
        // but only if it's actually working properly in this client.
        sensors.forEach(function(sensor) {
            if (!sensor.isWorking()) {
                return
            }

            var sensorData = {};
            var newData = sensor.getData();
            for (var key in newData) {
                if (newData.hasOwnProperty(key)) {
                    sensorData[key] = newData[key];
                }
            }

            var sensorName = 'atst.healthcheck.sensors.' + sensor.getName();
            var currentHash = hash(JSON.stringify(sensorData)).toString(36);
            var previousHash = localStorageWrapper.getWithDefaultOnError(sensorName, currentHash);
            if (previousHash===currentHash && Math.random()>0.01) {
                return;
            }
            // What's one more analytics event to the world? A drop in the data ocean.
            AJS.trigger('analytics', {
                name: sensorName,
                data: sensorData
            });
            localStorageWrapper.setItemQuietly(sensorName, currentHash);
        });

    });
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.troubleshooting.plugin-confluence:atst-healthcheck-sensors', location = 'js/sensors/page-protocols.js' */
define('troubleshooting/sensors/page-protocols', ['troubleshooting/ajs'], function(AJS) {
    var baseUrl = AJS.contextPath();
    var hasPerfData = window.performance && typeof window.performance.getEntriesByType === 'function';

    // WARNING: This is a rough assumption based on how the WRM works.
    // Consuming this from the WRM would be a better option, but meh.
    // See: https://bitbucket.org/atlassian/atlassian-plugins-webresource/src/master/atlassian-plugins-webresource/src/main/java/com/atlassian/plugin/webresource/WebResourceUrlProviderImpl.java
    // See also: https://stash.atlassian.com/projects/CP/repos/static-assets-url/browse/src/main/resources/ui/health-checks/health-checks.js
    var WRM_STATIC_ASSET_REGEX = new RegExp(baseUrl + '/s/.+?/_/');

    function isStaticResource(resource) {
        return WRM_STATIC_ASSET_REGEX.test(resource.name);
    }

    function getNextHopData(resource) {
        return (resource && resource.nextHopProtocol) || 'unknown';
    }

    function unique(elem, pos, arr) {
        return arr.indexOf(elem) === pos;
    }

    return {
        isWorking: function() {
            return hasPerfData;
        },
        getName: function() {
          return 'page-protocols';
        },
        getData: function() {
            var resources = window.performance.getEntriesByType('resource');
            var navigation = window.performance.getEntriesByType('navigation');
            var resourceProtocols = resources
                .filter(isStaticResource)
                .map(getNextHopData)
                .filter(unique)
                .sort();
            if (!resourceProtocols.length) {
                resourceProtocols.push('unknown');
            }
            return {
                resourceProtocols: resourceProtocols,
                navigationProtocol: getNextHopData(navigation[0]),
                userAgent: navigator.getUserAgent && "use-js-client-hints" || navigator.userAgent
            };
        }
    }
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.resolution.atlasplugins.samlsso.Confluence:com.resolution.atlasplugins.samlsso.general-resources', location = 'web/inject_general_confluence.js' */
AJS.$(function() {
    (function() {
        if(!window.resolution) {
            window.resolution = {};
        }
        var webSudoWithSAML = {};
        window.resolution.webSudoWithSAML = webSudoWithSAML;

        // Inject inject_websudo.js if we're on the WebSudo-Page
        webSudoWithSAML.loginForm = AJS.$('form[name="authenticateform"]');
        if(webSudoWithSAML.loginForm .length) {
            var formAction = webSudoWithSAML.loginForm .attr('action');
            if(formAction.includes('doauthenticate.action')) {
                webSudoWithSAML.webSudoDestination = webSudoWithSAML.loginForm.find('input[name="destination"]').val();
                webSudoWithSAML.injectDiv = AJS.$("<div>");
                webSudoWithSAML.loginForm.before(webSudoWithSAML.injectDiv);
                WRM.require("wr!com.resolution.atlasplugins.samlsso.Confluence:injectWebsudoResources");
            }
        }
    })();
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-page-hierarchy-plugin:confluence-copy-page-hierarchy-plugin-loader', location = 'js/loader.js' */
/**
 * Async resources loader for copy-page-hierarchy resources.
 */
require([
    'ajs',
    'wrm',
    'confluence/legacy'
], function (AJS, WRM, Confluence) {
    var COPY_CONTEXT = 'confluence-copy-page-hierarchy';
    var DELETE_CONTEXT = 'confluence-delete-page-hierarchy';

    AJS.toInit(function ($) {
        var $toolsCopyLink = $('#action-copy-page-link');
        var $operationsCopyLink = $('#copyPageLink');
        var $toolsDeleteLink = $('#action-remove-content-link');

        _loadResourcesAsync($toolsCopyLink, COPY_CONTEXT);
        _loadResourcesAsync($operationsCopyLink, COPY_CONTEXT);
        _loadResourcesAsync($toolsDeleteLink, DELETE_CONTEXT);
        _checkInProgressDeleteFlow($toolsDeleteLink);
    });

    /**
     * Loads the require resources for either copy or delete
     * @param {jQuery} $button The button that loads the resources
     * @param {string} context The WRM resource context to load
     * @returns {undefined}
     * @private
     */
    function _loadResourcesAsync($button, context) {
        if ($button.length) {
            $button.off('click');
            $button.on('click', function (e) {
                e.preventDefault();
                _loadResources($button, context);
            });
        }
    }

    /**
     * Function handler to run when the delete or copy button is clicked
     * @param {jQuery} $button The button that loads the resources
     * @param {string} context The WRM resource context to load
     * @returns {undefined}
     * @private
     */
    function _loadResources($button, context) {
        var $body = $('body');
        $button.off('click');

        var loadingIndicator = Confluence.PageLoadingIndicator($body);
        loadingIndicator.show();

        AJS.debug('Loading ' + context + ' resources...');
        WRM.require(['wrc!' + context], function () {
            loadingIndicator.hide();
            $button.addClass('confluencePageHierarchy');
            AJS.debug('Loaded ' + context + ' resources.');
        });
    }

    /**
     * Checks to see if there is a flow in progress based on url parameters
     * and loads the resources if the flow is in progress.
     * @param {jQuery} $toolsDeleteLink The Delete button, if present
     * @returns {undefined}
     * @private
     */
    function _checkInProgressDeleteFlow($toolsDeleteLink) {
        var hash = window.location.hash;
        if (hash.indexOf('delete-complete') === 1) {
            _loadResources($toolsDeleteLink, DELETE_CONTEXT);
        }
    }
});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.extra.jira:jirachart-resources', location = '/jirachart/jirachart.js' */
AJS.toInit(function(){var c=require("confluence/form-state-control");AJS.$(".jira-chart-macro-img").load(function(a){AJS.log("Jira Chart Macro - chart image loaded");c.enableElement(AJS.$(".insert-jira-chart-macro-button",window.parent.document))}).error(function(a){AJS.log("Jira Chart Macro - chart image loaded error");c.disableElement(AJS.$(".insert-jira-chart-macro-button",window.parent.document));var b=AJS.$(a.target).parent();a=b.parent();b.remove();b="Unable to render Jira chart macro due to an execution error.";
AJS.messages.error(a,{body:b})})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.extra.jira:jirachart-resources', location = '/jirachart/twodimensionalchart-showlink.js' */
var TwoDimensionalShowLink=function(a){var g=function(b){var d=a("#two-dimensional-chart-"+b),c=d.position();a("\x3cdiv /\x3e",{id:"twodimensional-dark-layout-"+b,"class":"jim-sortable-dark-layout",css:{top:c.top+"px",left:c.left+"px",width:d.width()+"px",height:d.height()+"px"}}).appendTo(d.parent())},e=function(b){a("#twodimensional-dark-layout-"+b).remove()},k=function(){var b=a(this).attr("data-chart-id");g(b);var d={pageId:a("#chart-page-id-"+b).val(),wikiMarkup:a("#chart-wiki-"+b).val(),isShowMore:a(this).attr("data-is-show-more")};
AJS.$.ajax({type:"POST",dataType:"html",url:Confluence.getContextPath()+"/plugins/servlet/twoDimensionalShowMoreRenderer",data:d,success:function(c){if(a(c).find(".aui-message.error,.aui-message.aui-message-error").length)c=a(c).find(".message").text(),a("#two-dimensional-chart-"+b).find(".show-error").html(c);else{var h=a(c).find(".show-link-container a").attr("data-chart-id");a("#two-dimensional-chart-"+b).replaceWith(c);f(h)}e(b)},error:function(){a("#two-dimensional-chart-"+b).find(".show-error").html("Unable to render Jira chart macro. Execution has timed out.");
e(b)}})},f=function(b){a("#show-link-"+b).on("click",k)};return{init:function(){a(".show-link-container a").each(function(){f(a(this).attr("data-chart-id"))})}}}(AJS.$);AJS.$(function(){TwoDimensionalShowLink.init()});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-collaborative-editor-plugin:synchrony-util', location = '/js/synchrony-util.js' */
define("confluence-collaborative-editor-plugin/synchrony-util",["ajs","jquery","underscore","confluence/meta","confluence/legacy"],function(d,f,p,h,c){function e(a){var b=h.get(a);void 0===b&&(b=f('meta[name\x3d"ajs-'+a+'"]').attr("content"),void 0!==b?h.set(a,b):(console.error("Missing '"+a+"' metadata value"),k()&&d.trigger("editor.error.message",{messageKey:"collaborative-editor-load-failure",message:d.I18n.getText("editor.collaborative-editing.refresh-editor.error"),close:"manual"})));return b}
function n(){return e("synchrony-base-url").split(",")}function m(a,b,g){if(0===g.length||0===b.length)a.reject();else{var c=g[0];f.ajax({url:c+"/heartbeat",cache:!1}).done(function(d,e,k){200===k.status?(l=c,d=0<g.slice(1).length?"primary":"secondary",e=0<l.indexOf("synchrony-proxy")?"synchrony-proxy":"synchrony-direct",h.set("synchrony-connection-order",d),h.set("synchrony-connection-type",e),f.when.apply(this,b.map(function(a){return f.ajax({url:c+"/resources"+a,dataType:"script",cache:!0})})).then(function(){a.resolve()})):
m(a,b,g.slice(1))}).fail(function(){m(a,b,g.slice(1))})}}function k(){return d.Rte.getEditor()&&d.Rte.getEditor().initialized}var l;return{retrieveMetadata:e,getEntityId:function(){return"/"+e("synchrony-app-id")+"/confluence-"+c.getContentId()},synchronyReady:function(){return"0"!==c.getContentId()},getServiceUrl:function(){return(l?l:n()[0])+"/v1"},getXhrFallbackFlag:function(){return e("use-xhr-fallback")},getLatestRevisionWithAttr:function(a,b){return p.last(a.filter(function(a){return a.meta&&
a.meta[b]}))},hasRevisionType:function(a,b){return a.some(function(a){return a.meta&&a.meta.type===b})},hasRevisionTrigger:function(a,b){return a.some(function(a){return a.meta&&a.meta.trigger===b})},asArray:function(){return Array.prototype.slice.call(arguments)},time:function(a){c.debugTime&&c.debugTime(a)},timeEnd:function(a){c.debugTimeEnd&&c.debugTimeEnd(a)},loadScript:function(a){var b=f.Deferred(),c=n();m(b,a,c);return b.promise()},isEditorInitialised:k}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-collaborative-editor-plugin:synchrony-content', location = '/js/synchrony-content.js' */
define("confluence-collaborative-editor-plugin/synchrony-content","jquery underscore ajs confluence/meta confluence/legacy confluence-collaborative-editor-plugin/synchrony-util".split(" "),function(d,h,f,l,m,k){var e;return{getContent:function(a){if("limited"===a.syncRevSource)return{error:"limited-mode"};var b=a.title,c=a.editorContent;var e="\x3cbody data-title\x3d'"+h.escape(b)+"'\x3e"+c+"\x3c/body\x3e";var f=!!d("\x3cdiv\x3e"+c+"\x3c/div\x3e").find(".fatal-render-error").length,g=a.syncRev;return{title:b,
raw:c,html:e,error:f,confRev:a.confRev,syncRev:"dummy-sync-rev"===g?null:g,syncRevSource:a.syncRevSource}},isUnpublished:function(){return k.retrieveMetadata("new-page")},fixTinymceCaretContainer:function(a,b){var c;!(c=!a.childNodes.length)&&(c=1===a.childNodes.length)&&(c=a.firstChild&&a.firstChild.classList&&a.firstChild.classList.contains("synchrony-container")&&/^\s*$/.test(a.firstChild.textContent));c&&b.setContent("")},writeTitleToRootElement:function(){var a=d("#content-title").val();if(a!==
e){var b=frames.wysiwygTextarea_ifr;(void 0!==b.contentDocument?b.contentDocument:b.document).body.setAttribute("data-title",a);e=a}},readTitleFromRootElement:function(a){a.hasAttribute("data-title")&&(a=a.getAttribute("data-title"),a!==e&&(d("#content-title").val(a),e=a))},bindPostPasteFix:function(){d(document).bind("prePaste",function(){f.trigger("synchrony.stop",{id:"confluence.postpaste-fix"})}).bind("postPaste",function(){setTimeout(function(){f.trigger("synchrony.start",{id:"confluence.postpaste-fix"})},
0)})},isContentEmpty:function(a){return/^\s+$/.test(a)||0===d(d.parseHTML?d.parseHTML(a):a).text().trim().length}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.watch-button:watch-resources', location = 'js/watch-model.js' */
define("confluence-watch-button/watch-model",["ajs","backbone"],function(c,e){return e.Model.extend({saveSettings:function(b,a,e){this.trigger("request");var d=this;return c.safe.ajax({url:b,type:a?"POST":"DELETE",contentType:"application/json",dataType:"json",data:{}}).done(function(){d.set(e,a);d.trigger("sync",d)}).fail(function(){d.trigger("error")})},saveWatchPage:function(b){var a=c.contextPath()+"/rest/api/user/watch/content/"+this.get("pageId");return this.saveSettings(a,b,"watchingPage")},
saveWatchBlogs:function(b){var a=c.contextPath()+"/rest/api/user/watch/space/"+this.get("spaceKey")+"?contentType\x3dblogpost";return this.saveSettings(a,b,"watchingBlogs")},saveWatchSpace:function(b){var a=c.contextPath()+"/rest/api/user/watch/space/"+this.get("spaceKey");return this.saveSettings(a,b,"watchingSpace")}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.watch-button:watch-resources', location = 'js/watch-view.js' */
define("confluence-watch-button/watch-view",["jquery","ajs","underscore","backbone","confluence/flag"],function(e,b,g,h,k){return h.View.extend({events:{"change #cw-watch-page":"changeWatchPage","change #cw-watch-blogs":"changeWatchBlogs","change #cw-watch-space":"changeWatchSpace"},initialize:function(){g.bindAll(this,"render","initTooltips","changeWatchPage","changeWatchBlogs","changeWatchSpace","togglePageEnabledState","toggleBlogsEnabledState","startLoading","stopLoading","setTitle","_restoreCheckboxState",
"_disableAllElements","_rememberLastState","_revertToLastState");this.model.on("sync change:watchingSpace",this.togglePageEnabledState,this);this.model.on("change:watchingSpace",this.toggleBlogsEnabledState,this);this.model.on("request",this.startLoading,this);this.model.on("sync",this.setTitle,this);this.model.on("sync",this.stopLoading,this);this.model.on("sync",this._restoreCheckboxState,this);this.model.on("error",this._revertToLastState,this)},render:function(){this.$el.html(Confluence.Watch.Templates.dialogBody(this.model.toJSON()));
this.initTooltips();this.setTitle(this.model);"READ_ONLY"===b.Meta.get("access-mode")&&this._disableAllElements();return this},_revertToLastState:function(){var a=this.currentModel.get("watchingPage"),c=this.currentModel.get("watchingSpace"),f=this.currentModel.get("watchingBlogs"),d=this.$("#cw-watch-page");d.prop("checked",a||c);d.prop("disabled",this.currentModel.get("watchPageDisabled"));d=this.$("#cw-watch-space");d.prop("checked",c);d.prop("disabled",this.currentModel.get("watchSpaceDisabled"));
d=this.$("#cw-watch-blogs");d.prop("checked",f);d.prop("disabled",this.currentModel.get("watchBlogsDisabled"));this.$(".cw-manage-watchers").removeClass("disabled");this.model.set("watchingPage",a);this.model.set("watchingBlogs",f);this.model.set("watchingSpace",c);this.stopLoading();this.errorFlag=new k(e.extend({},{close:"manual",type:"error",extraClasses:"confluence-menu-flag",fifo:!0,stack:"menu"},{body:"This site is read-only. You can\u0027t make changes right now."}))},_rememberLastState:function(){this.errorFlag&&
this.errorFlag.close();this.currentModel=this.model.clone();this.currentModel.set("watchPageDisabled",this.$("#cw-watch-page").prop("disabled"));this.currentModel.set("watchSpaceDisabled",this.$("#cw-watch-space").prop("disabled"));this.currentModel.set("watchBlogsDisabled",this.$("#cw-watch-blogs").prop("disabled"));this.$targetCheckbox.prop("disabled",!0)},_disableAllElements:function(){this.$("input[type\x3d'checkbox']").prop("disabled",!0);this.$(".cw-manage-watchers").addClass("disabled")},_restoreCheckboxState:function(){this.$targetCheckbox&&
this.$targetCheckbox.prop("disabled",!1)},initTooltips:function(){this.$(".cw-tooltip").tooltip({gravity:"e",offset:5,delayIn:0});this.togglePageEnabledState(this.model);this.toggleBlogsEnabledState(this.model)},changeWatchPage:function(a){this.$targetCheckbox=e(a.target);this._rememberLastState();a=this.$targetCheckbox.is(":checked");this.model.saveWatchPage(a)},changeWatchBlogs:function(a){this.$targetCheckbox=e(a.target);this._rememberLastState();a=this.$targetCheckbox.is(":checked");this.model.saveWatchBlogs(a)},
changeWatchSpace:function(a){this.$targetCheckbox=e(a.target);this._rememberLastState();a=this.$targetCheckbox.is(":checked");this.model.saveWatchSpace(a)},togglePageEnabledState:function(a){var c=a.get("watchingPage");a=a.get("watchingSpace");this.$("#cw-watch-page").prop("disabled",a);this.$("#cw-watch-page").prop("checked",c||a);c=a?"You will receive updates for this page because you are watching this space.":"";this.$(".cw-tooltip-watch-page").attr("original-title",c)},toggleBlogsEnabledState:function(a){var c=a.get("watchingBlogs");
a=a.get("watchingSpace");this.$("#cw-watch-blogs").prop("disabled",a);this.$("#cw-watch-blogs").prop("checked",c||a);c=a?"You are subscribed to all blog posts because you are watching this space.":"";this.$(".cw-tooltip-watch-blogs").attr("original-title",c)},startLoading:function(){this.$(".cw-status").addClass("loading")},stopLoading:function(){this.$(".cw-status").removeClass("loading")},setTitle:function(){var a=this.model.get("watchingPage"),c=this.model.get("watchingBlogs"),e=this.model.get("watchingSpace"),d=
this.model.get("isBlogPost");a=e?{title:"You are watching this space",description:"Receiving email updates for all content in this space."}:a&&d&&c?{title:"You are watching this blog post",description:"Receiving email updates about changes to this blog post and all new blog posts in this space."}:a&&d?{title:"You are watching this blog post",description:"Receiving email updates about changes to this blog post."}:a?{title:"You are watching this page",
description:"Receiving email updates about changes to this page."}:d&&c?{title:"You are watching for new blog posts",description:"Receiving email updates for new blog posts in this space."}:d?{title:"You are not watching this blog",description:"Start watching to receive email updates about changes to this blog."}:{title:"You are not watching this page",description:"Start watching to receive email updates about changes to this page."};
this.$(".cw-title").text(a.title);this.$(".cw-title-description").text(a.description)}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.watch-button:watch-resources', location = 'js/notification.js' */
define("confluence-watch-button/watch-notification",["jquery","confluence/flag"],function(d,b){return function(c){var a=document.getElementById("watch-notification");null!=a&&a.close();b({type:"success",body:c,close:"auto"}).setAttribute("id","watch-notification")}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.watch-button:watch-resources', location = 'js/watch.js' */
require("jquery underscore ajs confluence-watch-button/watch-model confluence-watch-button/watch-view confluence-watch-button/watch-notification".split(" "),function(e,k,d,l,m,n){function p(b,c){h(b,c);var a=new l(c),f=new m({model:a});d.InlineDialog(b,"confluence-watch",function(a,d,b){f.setElement(a);f.render();b()},{width:325,offsetX:-180,cacheContent:!1,hideDelay:null,hideCallback:function(){e(".tipsy").hide()}});a.on("change:watchingPage change:watchingBlogs change:watchingSpace",function(a){h(b,
a.toJSON())});a.on("change:watchingPage",function(a,b){d.trigger("analytics",{name:b?"watch-page":"unwatch-page"})});a.on("change:watchingBlogs",function(a,b){d.trigger("analytics",{name:b?"watch-blogs":"unwatch-blogs"})});a.on("change:watchingSpace",function(a,b){d.trigger("analytics",{name:b?"watch-space":"unwatch-space"})});q(a);var g=!1;e(document).on("keyup",function(){g=!1});window.CW_watchPage=function(){if(!g){g=!0;var c=a.get("watchingSpace"),f=a.get("watchingPage");c?e("body, #splitter-content").animate({scrollTop:0},
300,function(){b.click();setTimeout(function(){e(".cw-tooltip-watch-page").tipsy("show")},50)}):(c=!f,a.saveWatchPage(c),c=c?"You started watching this page.":"You stopped watching this page.",n(c))}}}function h(b,c){var a=c.isBlogPost&&c.watchingBlogs,e=c.watchingSpace;c.watchingPage||a||e?(c=b.find(".aui-icon").removeClass("aui-iconfont-unwatch").addClass("aui-iconfont-watch"),a=d.format("{0}W{1}atching","\x3cu\x3e",
"\x3c/u\x3e"),b.prop("title","Stop watching (w)").children("span").empty().append(c).append(" "+a)):(c=b.find(".aui-icon").removeClass("aui-iconfont-watch").addClass("aui-iconfont-unwatch"),a=d.format("{0}W{1}atch","\x3cu\x3e","\x3c/u\x3e"),b.prop("title","Watch (w)").children("span").empty().append(c).append(" "+a))}function q(b){b.on("change:watchingPage",function(b,a){d.trigger(a?"watchpage.pageoperation":
"unwatchpage.pageoperation")})}d.toInit(function(){var b=e("#watch-content-button");if(b.length){b.click(function(a){a.preventDefault()});var c=d.Meta.get("page-id");e.getJSON(d.contextPath()+"/rest/watch-button/1.0/watchState/"+c,function(a){k.extend(a,{pageId:c,spaceKey:d.Meta.get("space-key"),spaceName:d.Meta.get("space-name")});p(b,a);b.addClass("watch-state-initialised")})}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.watch-button:watch-resources', location = 'templates/watch.soy' */
// This file was automatically generated from watch.soy.
// Please don't edit this file by hand.

/**
 * @fileoverview Templates in namespace Confluence.Watch.Templates.
 */

if (typeof Confluence == 'undefined') { var Confluence = {}; }
if (typeof Confluence.Watch == 'undefined') { Confluence.Watch = {}; }
if (typeof Confluence.Watch.Templates == 'undefined') { Confluence.Watch.Templates = {}; }


Confluence.Watch.Templates.dialogBody = function(opt_data, opt_ignored) {
  return '<div class="cw-status"><h2 class="cw-title"></h2><p class="cw-title-description"></p></div><form class="aui cw-dialog"><div class="cw-tooltip cw-tooltip-watch-page"><div class="checkbox"><input class="checkbox" type="checkbox" id="cw-watch-page" ' + ((opt_data.watchingPage) ? 'checked' : '') + '><label for="cw-watch-page">' + ((opt_data.isBlogPost) ? soy.$$escapeHtml('Watch blog post') : soy.$$escapeHtml('Watch page')) + '</label></div></div>' + ((opt_data.isBlogPost) ? '<div class="cw-tooltip cw-tooltip-watch-blogs"><div class="checkbox"><input class="checkbox" type="checkbox" id="cw-watch-blogs" ' + ((opt_data.watchingBlogs) ? 'checked' : '') + '><label for="cw-watch-blogs">' + soy.$$escapeHtml('Watch for new blog posts in this space') + '</label></div></div>' : '') + '<div class="checkbox"><input class="checkbox" type="checkbox" id="cw-watch-space" ' + ((opt_data.watchingSpace) ? 'checked' : '') + '><label for="cw-watch-space">' + soy.$$escapeHtml('Watch all content in this space') + '</label></div></form>' + ((opt_data.isAdmin) ? '<div class="cw-manage-watchers-wrapper"><button class="aui-button aui-button-link cw-manage-watchers">' + soy.$$escapeHtml('Manage Watchers') + '</button></div>' : '');
};
if (goog.DEBUG) {
  Confluence.Watch.Templates.dialogBody.soyTemplateName = 'Confluence.Watch.Templates.dialogBody';
}

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.share-page:share-dialog-service', location = 'js/service/share-page.js' */
define("confluence/share-page/service/share-page",["ajs","wrm","jquery"],function(k,e,c){return{initDialog:function(d,f,g,a){var b=c(d);if(b.length)b.off("click.share-page").on("click.share-page",function(h){b.off("click.share-page");a.beforeLoad&&"function"===typeof a.beforeLoad&&a.beforeLoad();e.require(["wrc!share-page-async-loader"]).done(function(){a.afterLoad&&"function"===typeof a.afterLoad&&a.afterLoad();require("confluence/share-page/dialog-async-loader")(d,f,c.extend(!1,{hideCallback:function(){a.onHide&&
"function"===typeof a.onHide&&a.onHide()},width:250,hideDelay:36E5},g),a);b.click()});h.preventDefault();return!1})}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.share-page:share-dialog-service', location = 'js/service/share-dialog-loader.js' */
define("confluence/share-page/service/share-dialog-loader",["wrm","confluence/api/logger"],function(a,b){return{init:function(c,d){a.require(["wrc!share-dialog-react"]).done(function(){require(["share-dialog-react"],function(a){a.init(c,d)})}).fail(function(){b.error("fail to load share-dialog-app")})}}});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.share-page:view-resources', location = 'js/view-init.js' */
require(["ajs","confluence/meta","confluence/share-page/service/share-page","confluence/share-page/service/share-dialog-loader","confluence/dark-features"],function(c,a,e,f,g){var h="page"===a.get("content-type")?"Share this page":"Share this blog post",k="page"===a.get("content-type")?"Take a look at this page":"Take a look at this blog post";c.toInit(function(b){if(g.isEnabled("react.share.dialog"))f.init("share-react-link");
else{var d=b("#shareContentLink");b("#shareContentLink").length&&e.initDialog("#shareContentLink","shareContentPopup",{},{heading:h,notePlaceholder:k,link:function(){return b('link[rel\x3d"shortlink"]').prop("href")},entityId:function(){return a.get("page-id")},restriction:function(){return require("confluence/share-page/fetch/content-restrictions")(a.get("content-id")).pipe(function(a){var b={};if(a.read.restrictions.user.size||a.read.restrictions.group.size||a.update.restrictions.user.size||a.update.restrictions.group.size)b.type=
"restrict",b.message="Restrictions on this page may prevent people viewing or editing.";return b})},copyOption:"share",shareType:"view",contentType:a.get("content-type"),errorText:"An unexpected error occurred. Please try again.",beforeLoad:function(){d.find(".aui-icon").css("visibility","hidden");d.spin()},afterLoad:function(){d.find(".aui-icon").css("visibility","visible");d.spinStop()},onHide:function(){b(".dashboard-actions .explanation").hide()}})}})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.editor.confluence-source-editor:atlassian-source-editor-view-storage-javascript', location = 'jscripts/source-editor/view-source.js' */
/*
 * Copyright © 2012 - 2013 Atlassian Corporation Pty Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

'use strict';

require([
    'ajs',
    'jquery'
], function (
    AJS,
    $
) {
    AJS.toInit(function () {
        if ($('#action-view-storage-link').length) {
            // "View Storage Format" link - remove the one we added via a web item
            $('#action-source-editor-view-storage-link').closest('li').hide();
        }
    });
});

}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.web.resources:page-analytics', location = '/includes/js/page.js' */
define("confluence/page",["ajs","jquery","confluence/analytics-support","window","document"],function(a,c,e,f,g){var i=a.Meta.get("page-id"),d=function(b,a){e.publish("confluence.page."+b,c.extend({pageID:i},a||{}))},j=(new Date).getTime();return function(){var b=c("#main-content");if(0!==b.length){d("view");b.on("click","a",function(a){a=-1<a.currentTarget.href.indexOf(f.location.host)?"internal":"external";d("link.click",{linkType:a})});var h=c(f),e=(new Date).getTime();h.on("scroll.content",a.debounce(function(){var a=
g.body.scrollTop+g.body.offsetHeight,c=b.offset().top+b.height();a>c&&(d("scroll-to-bottom",{secondsSinceReadyEvent:((new Date).getTime()-e)/1E3,secondsSincePageLoad:((new Date).getTime()-j)/1E3}),h.off("scroll.content"))},200));setTimeout(function(){d("reading")},3E5)}}});require("confluence/module-exporter").safeRequire("confluence/page",function(a){require("ajs").toInit(a)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.macros.advanced:blog-post-resources', location = 'com/atlassian/confluence/plugins/macros/advanced/blog-posts.js' */
jQuery(function(a){a(".macro-blank-experience .create-post").bind("click",function(){var a=AJS.Meta.get("base-url")+"/pages/createblogpost.action?spaceKey\x3d"+AJS.Meta.get("space-key");window.location=a})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'confluence.macros.multimedia:flash-autosize', location = 'javascript/flash-autosize.js' */
AJS.toInit(function(b){function a(e,d){var c;if(d>=20){AJS.log("unable to auto size flash - took too long to load");return}c=b([]);e.each(function(){var g=b(this);var i;var f;if(this.GetVariable){if(!g.attr("height")){i=this.GetVariable("height");if(i){g.height(i)}else{c=c.add(g);return}}if(!g.attr("width")){f=this.GetVariable("width");if(f){g.width(f)}else{c=c.add(g);return}}}});if(c.length){setTimeout(function(){a(c,d+1)},100)}}a(b('embed[type="application/x-shockwave-flash"]'),0);b(window).bind("render-content-loaded",function(d,c){a(b('embed[type="application/x-shockwave-flash"]',c),0)})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.view-source:view-source-menu-resources', location = 'com/atlassian/confluence/plugins/viewsource/js/viewsource.js' */
define("confluence-view-source/viewsource",["jquery","window"],function(a,b){return function(){a("#action-view-source-link").click(function(a){b.open(this.href,(this.id+"-popupwindow").replace(/-/g,"_"),"width=800, height=600, scrollbars, resizable").opener=null;a.preventDefault();return!1})}});require("confluence/module-exporter").safeRequire("confluence-view-source/viewsource",function(a){require("ajs").toInit(a)});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.view-storage-format:view-storage-menu-resources', location = 'com/atlassian/confluence/plugins/viewstorage/js/viewstorage.js' */
AJS.toInit(function(a){a(".view-storage-link, .view-storage-link a").click(function(a){window.open(this.href,(this.id+"-popupwindow").replace(/-/g,"_"),"width\x3d800, height\x3d600, scrollbars, resizable");a.preventDefault();return!1})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.confluence-request-access-plugin:confluence-request-access-plugin-loader', location = '/js/loader.js' */
require(["ajs","wrm"],function(a,b){!a.$(".page-gadget").length&&a.toInit(function(){b.require(["wrc!request-access-plugin"])})});
}catch(e){WRMCB(e)};
;
try {
/* module-key = 'com.atlassian.confluence.plugins.quickedit:quick-comment-hide-traditional', location = 'jscripts/add-comment-hider.js' */
define("confluence-quick-edit/add-comment-hider",[],function(){return function(a){a("#comments-actions").hide()}});require("confluence/module-exporter").safeRequire("confluence-quick-edit/add-comment-hider",function(a){require("ajs").toInit(a)});
}catch(e){WRMCB(e)};