jQuery(document).ready(function($){
	var $lateral_menu_trigger = $('#cd-menu-trigger'),
		$content_wrapper = $('.cd-main-content'),
		$navigation = $('header');

	//打开关闭横向菜单单击菜单图标
	$lateral_menu_trigger.on('click', function(event){
		event.preventDefault();
		
		$lateral_menu_trigger.toggleClass('is-clicked');
		$navigation.toggleClass('lateral-menu-is-open');
		$content_wrapper.toggleClass('lateral-menu-is-open').one('webkitTransitionEnd otransitionend oTransitionEnd msTransitionEnd transitionend', function(){
			//firefox转换在父溢出更改时中断,因此我们需要等待转换结束以隐藏主体溢出
			$('body').toggleClass('overflow-hidden');
		});
		$('#cd-lateral-nav').toggleClass('lateral-menu-is-open');
	
		if($('html').hasClass('no-csstransitions')) {
			$('body').toggleClass('overflow-hidden');
		}
	});

	//关闭横向菜单单击菜单本身之外
	$content_wrapper.on('click', function(event){
		if( !$(event.target).is('#cd-menu-trigger, #cd-menu-trigger span') ) {
			$lateral_menu_trigger.removeClass('is-clicked');
			$navigation.removeClass('lateral-menu-is-open');
			$content_wrapper.removeClass('lateral-menu-is-open').one('webkitTransitionEnd otransitionend oTransitionEnd msTransitionEnd transitionend', function(){
				$('body').removeClass('overflow-hidden');
			});
			$('#cd-lateral-nav').removeClass('lateral-menu-is-open');
			//检查是否不支持转换
			if($('html').hasClass('no-csstransitions')) {
				$('body').removeClass('overflow-hidden');
			}

		}
	});

	//打开（或关闭）横向菜单中的子菜单项。关闭所有其他打开的子菜单项
	$('.item-has-children').children('a').on('click', function(event){
		event.preventDefault();
		$(this).toggleClass('submenu-open').next('.sub-menu').slideToggle(200).end().parent('.item-has-children').siblings('.item-has-children').children('a').removeClass('submenu-open').next('.sub-menu').slideUp(200);
	});
});